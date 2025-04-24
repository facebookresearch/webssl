# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dinov2.vision_transformer import (
    webssl_dino300m_full2b_224,
    webssl_dino1b_full2b_224,
    webssl_dino2b_full2b_224,
    webssl_dino2b_light2b_224,
    webssl_dino2b_heavy2b_224,
    webssl_dino3b_light2b_224,
    webssl_dino3b_heavy2b_224,
    webssl_dino3b_full2b_224,
    webssl_dino5b_full2b_224,
    webssl_dino7b_full8b_224,
    webssl_dino7b_full8b_378,
    webssl_dino7b_full8b_518
)
import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Model, ViTModel

# Load weights of the DINO teacher encoder
# The PyTorch state_dict is already preprocessed to have the right key names
def load_pretrained_dino_weights(model, pretrained_weights):
    state_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

# Adjust to your liking - e.g. the input resolution, and whether to crop / what crop resolution.
def build_pt_transform(img_size):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    # For this tutorial, we will omit center crop and resize directly to a square image. You may find what works best for you
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC), # resize shortest side to img_size
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    return eval_transform

def forward_dino(model_hf, model_pt):
    # Run a sample inference with DINO
    with torch.no_grad():
        # Read and pre-process the image
        im = Image.open('sample_images/bird.JPEG')
        x_pt = pt_transform(im).cuda().unsqueeze(0)
        x_hf = hf_transform(im, return_tensors="pt").to("cuda")
        # Extract the patch-wise features from the last layer
        out_patch_features_pt = model_pt.forward_features(x_pt)['x_norm_patchtokens'] # x_norm_clstoken is the cls token
        out_patch_features_hf = model_hf(**x_hf).last_hidden_state[:, 1:] # index 0 is the cls token

    return out_patch_features_hf, out_patch_features_pt

if __name__ == '__main__':
    model_name = 'webssl-dino1b-full2b-224' # Replace with your favored model, e.g. webssl-dino1b-full2b-224 or webssl-dino7b-full8b-518
    model_name_with_underscore = model_name.replace('-', '_')

    # HuggingFace model repo name
    hf_model_name = f'facebook/{model_name}'
    # Path to local PyTorch weights
    pt_model_path = f'YOUR_PATH_TO_TORCH_WEIGHTS.pth'

    # Initialize the HuggingFace model, load pretrained weights
    model_hf = Dinov2Model.from_pretrained(hf_model_name, attn_implementation='sdpa') # 'eager' mode also supported
    model_hf.cuda().eval()
    
    # Build HuggingFace preprocessing transform
    # For this tutorial, we will omit center crop and resize directly to a square image. You may find what works best for your use-case
    hf_transform = AutoImageProcessor.from_pretrained(hf_model_name, use_fast=False)
    hf_transform.do_center_crop = False
    hf_transform.size = {
        'height': hf_transform.size['shortest_edge'],
        'width': hf_transform.size['shortest_edge']
    }

    # Initialize the PyTorch model, load pretrained weights
    model_pt = globals()[model_name_with_underscore]() # fancy way to call a method given the name, e.g. webssl_dino7b_full8b_518()
    model_pt.cuda().eval()
    load_pretrained_dino_weights(model_pt, pt_model_path) # we won't load the dino_head and ibot_head from SSL
    
    # Build PyTorch preprocessing transform
    pt_transform = build_pt_transform(img_size = hf_transform.crop_size['height'])

    # Inference
    out_patch_features_hf, out_patch_features_pt = forward_dino(model_hf, model_pt)

    print(out_patch_features_hf.shape, out_patch_features_pt.shape, torch.abs(out_patch_features_pt - out_patch_features_hf).sum(), torch.allclose(out_patch_features_pt, out_patch_features_hf, atol=1e-3, rtol=1e-3))
