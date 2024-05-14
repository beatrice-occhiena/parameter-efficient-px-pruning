import open_clip
from sympy import root
import torch
import json
from tqdm import tqdm
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.models import vit_b_32
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100
from datasets.TinyImageNet.dataset import TINYIMAGENET
from torchvision import transforms
from argparse import ArgumentParser

from traitlets import default

# CMD ARGUMENTS #
parser = ArgumentParser()
parser.add_argument('--model', type=str, help='kaiming, torch or clip')
parser.add_argument('--dataset', type=str, help='cifar10, cifar100 or tiny', default=None)

# FUNCTIONS #

def kaiming_init(model):
  # Kaiming initialization
  for name, param in model.named_parameters():
    if "ln_" in name: # Layer Normalization
      if "weight" in name:
        nn.init.constant_(param, 1)
      if "bias" in name:
        nn.init.constant_(param, 0)
    elif "weight" in name:
        if "attn" in name: # Attention weights
          nn.init.xavier_uniform_(param)
        else:
          nn.init.kaiming_normal_(param)
    elif "bias" in name:
      nn.init.constant_(param, 0)
    elif "class" in name: # Class Embedding
      nn.init.normal_(param, mean=0, std=0.01)  # small random values
    else:
      nn.init.kaiming_normal_(param)

def vitB32_init_kaiming():
  # VIT-B/32 INITIALIZED from scratch with Kaiming init
  # model = vit_b_32()
  # kaiming_init(model)
  # return (model, None, None)
  clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
  vision_model = clip_model.visual
  kaiming_init(vision_model)
  return (vision_model, preprocess_train, preprocess_val)

def vitB32_pretrained_torch():
  # VIT-B/32 PRETRAINED from standard torchvision.models
  model = vit_b_32(weights='IMAGENET1K_V1')
  return (model, None, None)

def vitB32_pretrained_clip():
  # VIT-B/32 PRETRAINED from mlfoundations/open_clip
  clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
  vision_model = clip_model.visual
  return (vision_model, preprocess_train, preprocess_val)
  
def save_weights_ranks(model, file_name, check_grad=False):
  weights_rank = {}
  for name, param in model.named_parameters():

    if check_grad:
      param = param.grad

    if param.dim() == 3: # In standard ViT from torchvision
      param = param.squeeze()
    
    if param.dim() == 2: # Standard weight matrix
      size = list(param.size())
      rank = torch.linalg.matrix_rank(param).item()
    
    elif param.dim() == 4: # Convolutional weight matrix
      # Reshape to copute the rank of DeltaW as in LoRA
      out_ch, in_ch, k_size, _ = param.size()
      reshaped_param = param.view(out_ch * k_size, in_ch * k_size)
      size = list(reshaped_param.size())
      rank = torch.linalg.matrix_rank(reshaped_param).item()
    
    else: # One dimensional weight tensors -> no rank
      continue

    weights_rank[name] = {"size": size, "rank": rank}

  with open(f"{file_name}.json", 'w') as fp:
    json.dump(weights_rank, fp)

def load_cifar10(preprocess_train):  
  # Load CIFAR10 dataset & transform the images
  """
  official openai class PreprocessCfg for data transformation:
    size: Union[int, Tuple[int, int]] = 224
    mode: str = 'RGB'
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = 'bicubic'
    resize_mode: str = 'shortest'
    fill_color: int = 0
  """
  train_data = CIFAR10(root='./data', train=True, download=True, transform=preprocess_train)
  train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
  num_classes = 10

  return num_classes, train_loader

def load_cifar100(preprocess_train):  
  # Load CIFAR100 dataset & transform the images
  """
  official openai class PreprocessCfg for data transformation:
    size: Union[int, Tuple[int, int]] = 224
    mode: str = 'RGB'
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = 'bicubic'
    resize_mode: str = 'shortest'
    fill_color: int = 0
  """
  train_data = CIFAR100(root='./data', train=True, download=True, transform=preprocess_train)
  train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
  num_classes = 100

  return num_classes, train_loader

def load_tinyImageNet(preprocess_train):
  # Load Tiny-ImageNet & transform the images
  """
  official openai class PreprocessCfg for data transformation:
    size: Union[int, Tuple[int, int]] = 224
    mode: str = 'RGB'
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = 'bicubic'
    resize_mode: str = 'shortest'
    fill_color: int = 0
  """
  train_data = TINYIMAGENET(root='./data', train=True, download=True, transform=preprocess_train)
  train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
  num_classes = 200

  return num_classes, train_loader
  
def compute_grad_ranks(vision_model, num_classes, args):

  # New classification head: 512 out_features of vitB32_clip -> N classes of DATASET
  class_embedding_size = vision_model.output_dim
  head = nn.Linear(class_embedding_size, num_classes)
  kaiming_init(head)
  model = nn.Sequential(vision_model, head)
  model = model.to('cuda')

  # Loss function
  loss_fn = nn.CrossEntropyLoss()

  # Check the rank of the gradient matrices after 1 forward pass of all data in the dataset
  for data, labels in tqdm(train_loader):
    data = data.to('cuda')
    labels = labels.to('cuda')
    
    # FORWARD PASS ->
    logits = model(data)
    loss = loss_fn(logits, labels)
    # BACKWARD PASS <-
    loss.backward()

  # Save the rank of the gradient matrices
  save_weights_ranks(model, f'ranks/vitB32_grad_rank_{args.model}_{args.dataset}', check_grad=True)

# MAPPINGS #
models = {
  'torch': vitB32_pretrained_torch,
  'kaiming': vitB32_init_kaiming,
  'clip': vitB32_pretrained_clip
}
datasets = {
  'cifar10': load_cifar10,
  'cifar100': load_cifar100,
  'tiny': load_tinyImageNet
}

# EXECUTE #
args = parser.parse_args()

# 1. Select the ViT-B/32 model
if args.model in models:
  model, preprocess_train, preprocess_val = models[args.model]()
else:
  print(f"Model {args.model} not recognized. Please choose from 'kaiming', 'torch', or 'clip'.")
  exit(-1)

# 2. Select the type of matrices for the rank computation
if not args.dataset: 
  # 2A. Save the ranks of the weight matrices
  save_weights_ranks(model, f"ranks/vitB32_weights_rank_{args.model}")

else: 
  # 2B. Save the ranks of the gradient matrices

  # 2B.1. Selcet the dataset
  if args.dataset in datasets:
    num_classes, train_loader = datasets[args.dataset](preprocess_train)
  else:
    print(f"Dataset {args.dataset} not recognized. Please choose from 'cifar10', 'cifar100', or 'tiny'.")

  # 2B.2. Save the gradient after one forward and backwrad pass for each batch of the dataset
  compute_grad_ranks(model, num_classes, args)

print("file saved!")