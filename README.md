# Web-SSL: Scaling Language-Free Visual Representation Learning

Official inference code for the **Web-SSL** model family introduced in: [Scaling Language-Free Visual Representation Learning](https://arxiv.org/abs/2504.01017).

[David Fan](https://davidfan.io)\*, [Shengbang Tong](https://tsb0601.github.io/)\*, [Jiachen Zhu](https://jiachenzhu.github.io), [Koustuv Sinha](https://koustuvsinha.com/), [Zhuang Liu](https://liuzhuang13.github.io), [Xinlei Chen](https://xinleic.xyz/), [Michael Rabbat](https://scholar.google.com/citations?user=cMPKe9UAAAAJ), [Nicolas Ballas](https://scholar.google.com/citations?user=euUV4iUAAAAJ), [Yann LeCun](http://yann.lecun.com), [Amir Bar](https://www.amirbar.net/)†, [Saining Xie](https://www.sainingxie.com/)†

FAIR Meta, New York University, Princeton University  
\*equal contribution, †equal advising

[<img src="https://img.shields.io/badge/arXiv-2504.01017-b31b1b.svg" height="22">](https://arxiv.org/abs/2504.01017)
[<img src="https://img.shields.io/badge/Project-Page-blue" height="22">](https://davidfan.io/webssl/)
[<img src="https://img.shields.io/badge/🤗-Models-yellow" height="22">](https://huggingface.co/collections/facebook/web-ssl-68094132c15fbd7808d1e9bb)
<p align="center">
<img src="https://davidfan.io/webssl/assets/figures/fig1_simple_v2.png" width=75% height=75% 
class="center">
</p>


## Overview

Web-SSL explores the scaling potential of visual self-supervised learning (SSL) on web-scale data. By scaling model size and training data, we show that vision-only models can match and even surpass language-supervised methods like CLIP, challenging the prevailing assumption that language supervision is necessary to learn strong visual representations for multimodal modeling. We present Web-SSL: a family of vision-only models, ranging from 0.3B to 7B parameters, that offers a strong alternative to CLIP for both multimodal modeling and classic vision tasks.

Key findings:
- 📈 SSL improves continuously with both model capacity and data.
- 🔍 Web-SSL matches or exceeds language-supervised methods on a wide range of VQA tasks—even on language-related tasks like OCR & Chart understanding, which were traditionally dominated by CLIP.
- 🖼️ Our models maintain competitive performance on classic vision tasks like classification and segmentation while excelling at multimodal tasks.
- 📊 Visual SSL methods are sensitive to data distribution! Training on filtered datasets with a higher concentration of text-rich images substantially improves OCR & Chart understanding.

## Our Models
We provide our model weights in both HuggingFace and native PyTorch format. Please see the [Usage](#usage) section for sample model loading and inference code.

### Web-DINO Models

#### Standard Models

Web-DINO is a family of DINOv2 models ranging from 0.3B to 7B parameters trained on larger scale web images. Web-DINO models especially excel at multimodal tasks such as VQA, without sacrificing performance in classic vision tasks such as image classification. Please see our paper for full details.

| Model | Patch Size | Resolution | Data | HuggingFace | Weights |
|-------|------------|------------|------|-------------|---------|
| webssl-dino300m-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-dino300m-full2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino300m_full2b_224.pth) |
| webssl-dino1b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-dino1b-full2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino1b_full2b_224.pth) |
| webssl-dino2b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-dino2b-full2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_full2b_224.pth) |
| webssl-dino3b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-dino3b-full2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_full2b_224.pth) |
| webssl-dino5b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-dino5b-full2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino5b_full2b_224.pth) |
| **webssl-dino7b-full8b-224** ⭐ | 14x14 | 224x224 | 8B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-dino7b-full8b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_224.pth) |
| **webssl-dino7b-full8b-378** ⭐ | 14x14 | 378x378 | 8B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-dino7b-full8b-378) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_378.pth) |
| **webssl-dino7b-full8b-518** ⭐ | 14x14 | 518x518 | 8B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-dino7b-full8b-518) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_518.pth) |

**Model Notes:**
- **webssl-dino7b-full8b-224** ⭐: Best 224x224 resolution model
- **webssl-dino7b-full8b-378** ⭐: Better performance with 378x378 resolution
- **webssl-dino7b-full8b-518** ⭐: Best overall performance with 518x518 resolution

#### Filtered Data Models

These models were trained on filtered subsets of MC-2B images with a higher concentration of text (e.g. signs, charts, tables, annotations, etc). This enhances OCR & Chart understanding capabilities without a notable performance drop in other VQA categories, relative to same-size models trained on the full data.

| Model | Patch Size | Resolution | Data | HuggingFace | Weights |
|-------|------------|------------|------|-------------|---------|
| webssl-dino2b-light2b-224 | 14x14 | 224x224 | 2B (MC-2B light) | [Link](https://huggingface.co/facebook/webssl-dino2b-light2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_light2b_224.pth) |
| webssl-dino2b-heavy2b-224 | 14x14 | 224x224 | 2B (MC-2B heavy) | [Link](https://huggingface.co/facebook/webssl-dino2b-heavy2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_heavy2b_224.pth) |
| webssl-dino3b-light2b-224 | 14x14 | 224x224 | 2B (MC-2B light) | [Link](https://huggingface.co/facebook/webssl-dino3b-light2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_light2b_224.pth) |
| webssl-dino3b-heavy2b-224 | 14x14 | 224x224 | 2B (MC-2B heavy) | [Link](https://huggingface.co/facebook/webssl-dino3b-heavy2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_heavy2b_224.pth) |

**Data Notes:**
- **MC-2B light**: 50.3% subset of MC-2B images that contain text
- **MC-2B heavy**: 1.3% subset of MC-2B images that contain charts/documents

### Web-MAE Models

Web-MAE is a family of MAE models ranging from 0.3B to 3B parameters, trained on larger scale web images. We release only the encoder for feature extraction.

| Model | Patch Size | Resolution | Data | HuggingFace | Weights |
|-------|------------|------------|------|-------------|---------|
| webssl-mae300m-full2b-224 | 16x16 | 224x224 | 2B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-mae300m-full2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_mae300m_full2b_224.pth) |
| webssl-mae700m-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-mae700m-full2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_mae700m_full2b_224.pth) |
| webssl-mae1b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-mae1b-full2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_mae1b_full2b_224.pth) |
| webssl-mae2b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-mae2b-full2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_mae2b_full2b_224.pth) |
| webssl-mae3b-full2b-224 | 14x14 | 224x224 | 2B (MC-2B) | [Link](https://huggingface.co/facebook/webssl-mae3b-full2b-224) | [Link](https://dl.fbaipublicfiles.com/webssl/webssl_mae3b_full2b_224.pth) |


### Additional Models

In response to community feedback, we are open-sourcing a few add-ons that are not part of the official release. Please see [ADDITIONAL_MODELS.md](ADDITIONAL_MODELS.md) for more details.

## Installation
It is possible that older or newer versions will work. However, we haven't tested them for this inference code.

```
conda create -n webssl python=3.11
conda activate webssl
pip install torch==2.5.1 torchvision==0.20.1 xformers --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.48.0 huggingface-hub==0.27.1 timm==1.0.15
```

## Usage

We provide two examples to use our models with both HuggingFace and native PyTorch. Note that you are not limited to using the pretraining resolution for inference, however, you will probably get the best results by inferencing with the same resolution.

### 1. Using HuggingFace Transformers
You may choose to download the model weights locally first using [huggingface-cli](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli). This is convenient when you don't have a large cache or when the network is slow.

E.g. `huggingface-cli download facebook/webssl-dino7b-full8b-518 --local-dir YOUR_PATH`, then supply `YOUR_PATH` to `from_pretrained()`.

```python
from transformers import AutoImageProcessor, Dinov2Model

# Load a Web-DINO model
model_name = "facebook/webssl-dino1b-full2b-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = Dinov2Model.from_pretrained(model_name, attn_implementation='sdpa') # 'eager' attention also supported
model.cuda().eval()

# Process an image
from PIL import Image

image = Image.open("sample_images/bird.JPEG")
with torch.no_grad():
  inputs = processor(images=image, return_tensors="pt").to('cuda')
  outputs = model(**inputs)
  last_hidden_states = outputs.last_hidden_state
```

### 2. Using PyTorch with original weights

```python
from dinov2.vision_transformer import webssl_dino1b_full2b_224
import torch
from PIL import Image
from torchvision import transforms

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Load model
model = webssl_dino1b_full2b_224(pretrained=True)
model.cuda().eval()

# Process an image
image = Image.open("sample_images/bird.JPEG")
x = transform(image).unsqueeze(0).cuda()
with torch.no_grad():
    features = model.forward_features(x)
    patch_features = features['x_norm_patchtokens']
```

See [demo_webdino.py](demo_webdino.py) and [demo_webmae.py](demo_webmae.py) for a complete example comparing HuggingFace and PyTorch implementations.


## Citation

If you find this repository useful for your research, please consider citing:

```bibtex
@article{fan2025scaling,
  title={Scaling Language-Free Visual Representation Learning},
  author={Fan, David and Tong, Shengbang and Zhu, Jiachen and Sinha, Koustuv and Liu, Zhuang and Chen, Xinlei and Rabbat, Michael and Ballas, Nicolas and LeCun, Yann and Bar, Amir and others},
  journal={arXiv preprint arXiv:2504.01017},
  year={2025}
}
```

## License
The majority of Web-SSL is licensed under CC-BY-NC, however portions of the project are available under separate license terms: DINOv2 is licensed under the Apache 2.0 license.

## Acknowledgements

We thank the [DINOv2](https://github.com/facebookresearch/dinov2) and [MAE](https://github.com/facebookresearch/mae) teams for their excellent codebases, and the [MetaCLIP](https://github.com/facebookresearch/MetaCLIP) team for the wonderful MetaCLIP dataset and codebase. We thank the [Cambrian](https://github.com/cambrian-mllm/cambrian) team for their insightful study into the role of vision encoders in MLLMs and their evaluation suite. Lastly, we thank our amazing collaborators in FAIR and the Meta Open-Source team for making this possible.
