from lib.layers_vit import mask_pretrained_vit, Linear
import open_clip
import torch.nn as nn

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

def vitB32(num_classes, device, dtype):

  # VIT-B/32 PRETRAINED from mlfoundations/open_clip
  clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
  vision_model = clip_model.visual

  # Substitute all layers with maskable ones
  mask_pretrained_vit(vision_model, device, dtype)

  # New maskable classification head: 512 out_features of vitB32_clip -> N classes of DATASET
  class_embedding_size = vision_model.output_dim
  head = Linear(class_embedding_size, num_classes)
  kaiming_init(head)
  model = nn.Sequential(vision_model, head)

  print("ViT-B/32 correctly imported and assembled with classification head and maskable layers!")

  return model