# Additional Models for Web-SSL

[<img src="https://img.shields.io/badge/arXiv-2504.01017-b31b1b.svg" height="22">](https://arxiv.org/abs/2504.01017)
[<img src="https://img.shields.io/badge/Project-Page-blue" height="22">](https://davidfan.io/webssl/)
[<img src="https://img.shields.io/badge/🤗-Models-yellow" height="22">](https://huggingface.co/collections/facebook/)

In response to community feedback, we are open-sourcing a few additional models that are not part of the official release. Please note that we cannot accommodate every request.

## SSL Head Weights
We provide the original teacher checkpoints from SSL pretraining for those who want to examine the SSL heads or continue SSL pretraining with our models. We only provide PyTorch format for these models. Note that these models are exactly the same as the official model weights, except with the original checkpoint key names, and SSL heads included. See the main README for model and data notes.

To load the models for SSL, please refer to the original DINOv2 codebase, e.g. [this function](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/setup.py#L62) as a starting point. You will need to adjust your config to use the right model architecture.

| Model | Patch Size | Resolution | Data | SSL Weights |
|-------|------------|------------|------|---------|
| webssl-dino300m-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino300m_full2b_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino300m_full2b_224_ssl_heads.pth) |
| webssl-dino1b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino1b_full2b_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino1b_full2b_224_ssl_heads.pth) |
| webssl-dino2b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_full2b_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_full2b_224_ssl_heads.pth) |
| webssl-dino3b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_full2b_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_full2b_224_ssl_heads.pth) |
| webssl-dino5b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino5b_full2b_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino5b_full2b_224_ssl_heads.pth) |
| **webssl-dino7b-full8b-224** | 14x14 | 224x224 | 8B (MC-2B) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_224_ssl_heads.pth) |
| **webssl-dino7b-full8b-378** | 14x14 | 378x378 | 8B (MC-2B) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_378_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_378_ssl_heads.pth) |
| **webssl-dino7b-full8b-518** | 14x14 | 518x518 | 8B (MC-2B) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_518_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_518_ssl_heads.pth) |
| webssl-dino2b-light2b-224 | 14x14 | 224x224 | 2B (MC-2B light) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_light2b_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_light2b_224_ssl_heads.pth) |
| webssl-dino2b-heavy2b-224 | 14x14 | 224x224 | 2B (MC-2B heavy) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_heavy2b_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_heavy2b_224_ssl_heads.pth) |
| webssl-dino3b-light2b-224 | 14x14 | 224x224 | 2B (MC-2B light) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_light2b_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_light2b_224_ssl_heads.pth) |
| webssl-dino3b-heavy2b-224 | 14x14 | 224x224 | 2B (MC-2B heavy) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_heavy2b_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_heavy2b_224_ssl_heads.pth) |


## ImageNet-1K Pretrained Models
We provide the DINO ViT-1B to 3B models trained on ImageNet-1k in PyTorch format. We do not recommend using these models in practice, as they will not perform well compared to our official Web-SSL model family trained on web-scale images. Please also note that these Web-SSL variants were trained by us for the purpose of ablations, and are not related to the official DINO/v2/v3 releases.

| Model | Patch Size | Resolution | Data | Encoder Weights | SSL Weights |
|-------|------------|------------|------|-----------------|-------------|
| webssl-dino1b-in1k-224 | 14x14 | 224x224 | IN-1k | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino1b_in1k_224.pth) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino1b_in1k_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino1b_in1k_224_ssl_heads.pth) |
| webssl-dino2b-in1k-224 | 14x14 | 224x224 | IN-1k | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_in1k_224.pth) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_in1k_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_in1k_224_ssl_heads.pth) |
| webssl-dino3b-in1k-224 | 14x14 | 224x224 | IN-1k | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_in1k_224.pth) | [Full Teacher](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_in1k_224_ssl_teacher.pth) / [Heads Only](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_in1k_224_ssl_heads.pth) |