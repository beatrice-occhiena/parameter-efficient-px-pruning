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
from torchvision import transforms

def kaiming_init(model):
  # Kaiming initialization
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      nn.init.kaiming_normal_(m.weight)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)

def vitB32_weights_rank_kaiming():
  # VIT-B/32 INITIALIZED from scratch with Kaiming init
  model = vit_b_32()
  kaiming_init(model)
  save_weights_rank(model, "ranks/vitB32_weights_rank_kaiming")

def vitB32_weights_rank_torch():
  # VIT-B/32 PRETRAINED from standard torchvision.models
  model = vit_b_32(weights='IMAGENET1K_V1')
  save_weights_rank(model, "ranks/vitB32_weights_rank_torch")

def vitB32_weights_rank_clip():
  # VIT-B/32 PRETRAINED from mlfoundations/open_clip
  clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
  vision_model = clip_model.visual
  save_weights_rank(vision_model, "ranks/vitB32_weights_rank_clip")
  
  # cuda

def save_weights_rank(model, file_name, check_grad=False):
  weights_rank = {}
  for name, param in model.named_parameters():

    if check_grad:
      param = param.grad

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

def cifar10():
  # VIT-B/32 PRETRAINED from mlfoundations/open_clip
  clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
  vision_model = clip_model.visual
  #to CUDA!!!
  
  # Load CIFAR10 dataset & transform the images
  """
  official openai class PreprocessCfg:
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

  # New classification head: 512 out_features of vitB32 -> 10 classes of CIFAR10
  class_embedding_size = vision_model.output_dim
  head = nn.Linear(class_embedding_size, 10)
  kaiming_init(head)
  model = nn.Sequential(vision_model, head)

  # Loss function
  loss_fn = nn.CrossEntropyLoss()

  # Check the rank of the gradient matrices after 1 forward pass of all data in the dataset
  for data, labels in tqdm(train_loader):
    #data.to('cuda')
    #labels.to('cuda')
    
    # FORWARD PASS ->
    logits = model(data)
    loss = loss_fn(logits, labels)
    # BACKWARD PASS <-
    loss.backward()

  # Save the rank of the gradient matrices
  save_weights_rank(model, "ranks/vitB32_grad_rank_clip", check_grad=True)


    


# CODE #

cifar10()