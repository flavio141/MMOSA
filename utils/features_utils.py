import os
import PIL
import requests
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from torchvision import transforms

embed_sizes={"dinov2_vits14": 384,
             "dinov2_vitb14": 768,
             "dinov2_vitl14": 1024,
             "dinov2_vitg14": 1536}


def trasform_img(target_img_size=224, normalize = False):
    trsforms = []

    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size))
	
    trsforms.append(transforms.ToTensor())
    if normalize == True:
        trsforms.append(transforms.Normalize())
    trsforms = transforms.Compose(trsforms)

    return trsforms


def tensor_to_image(img, trsforms, normalize = False):
    tensor_predict = trsforms(img)

    if normalize == True:
        tensor = (tensor_predict - tensor_predict.min()) / (tensor_predict.max() - tensor_predict.min())
    
    image_array = np.transpose(tensor_predict.numpy(), (1, 2, 0))
    
    tensor = image_array*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor), tensor_predict


def get_dino_bloom(modelpath = "models/DinoBloom-L.pth", 
                   modelname = "dinov2_vitl14", 
                   url = "https://zenodo.org/records/10908163/files/DinoBloom-L.pth?download=1"):
    
    model = torch.hub.load('facebookresearch/dinov2', modelname)

    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists(modelpath):
        response = requests.get(url)
        with open(modelpath, 'wb') as f:
            f.write(response.content)
    
    pretrained = torch.load(modelpath, map_location=torch.device('cpu'))
    new_state_dict = {}

    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key or "ibot_head" in key:
            pass
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

    pos_embed = nn.Parameter(torch.zeros(1, 257, embed_sizes[modelname]))
    model.pos_embed = pos_embed

    model.load_state_dict(new_state_dict, strict=True)
    return model
