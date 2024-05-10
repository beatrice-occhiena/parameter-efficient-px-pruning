import open_clip
import torchvision
import torch
import json

# cuda
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
vision_model = model.visual
state_dict = vision_model.state_dict()


  
# VIT-B/32 PRETRAINED
weights_rank = {}
for name, param in vision_model.named_parameters():

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

with open('vitB32_pre_weights_rank.json', 'w') as fp:
  json.dump(weights_rank, fp)
