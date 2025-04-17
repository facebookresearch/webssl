# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dinov2.vision_transformer import vit_1b, vit_2b, vit_3b, vit_5b, vit_7b
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Model

def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

def build_pt_transform():
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    eval_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    return eval_transform

if __name__ == '__main__':
    hf_model_name = 'facebook/webssl-dino1b-full2b-224'
    pt_model_path = '/checkpoint/amaia/video/davidfan/experiments/dinov2_scaling/metaclipv2/vitg14/16_node/2b_iter/job_1617224/eval/training_649999/teacher_checkpoint.pth'
    kwargs = dict(
        patch_size=14,
        ffn_layer='swiglu',
        init_values=1.0e-05,
        block_chunks=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        interpolate_offset=0.1,
        interpolate_antialias=False
    )
    model_pt = vit_1b(**kwargs)
    load_pretrained_weights(model_pt, pt_model_path, 'teacher') # we won't load the dino_head and ibot_head from SSL
    model_pt.cuda().eval()
    pt_transform = build_pt_transform()

    model_hf = Dinov2Model.from_pretrained(hf_model_name)
    model_hf.cuda().eval()
    hf_transform = AutoImageProcessor.from_pretrained(hf_model_name)

    with torch.no_grad():
        im = Image.open('sample_images/bird.JPEG')
        # x = torch.rand(1, 3, 224, 224).cuda()
        x_pt = pt_transform(im).cuda().unsqueeze(0)
        x_hf = hf_transform(im, return_tensors="pt").to("cuda")
        out_patch_features_pt = model_pt.forward_features(x_pt)['x_norm_patchtokens']
        out_patch_features_hf = model_hf(**x_hf).last_hidden_state[:, 1:]

    print(out_patch_features_hf.shape, out_patch_features_pt.shape, torch.allclose(out_patch_features_pt, out_patch_features_hf, atol=1e-4, rtol=1e-4))
