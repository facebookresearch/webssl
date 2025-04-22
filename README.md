# Web-SSL: Scaling Language-Free Visual Representation Learning

Official inference code for the **Web-SSL** family of models introduced in our work: [Scaling Language-Free Visual Representation Learning](https://arxiv.org/abs/2504.01017).

[David Fan](https://davidfan.io)<sup>* </sup>, [Shengbang Tong](https://tsb0601.github.io/)<sup>*</sup>, [Jiachen Zhu](https://jiachenzhu.github.io), [Koustuv Sinha](https://koustuvsinha.com/), [Zhuang Liu](https://liuzhuang13.github.io), [Xinlei Chen](https://xinleic.xyz/), [Michael Rabbat](https://scholar.google.com/citations?user=cMPKe9UAAAAJ), [Nicolas Ballas](https://scholar.google.com/citations?user=euUV4iUAAAAJ), [Yann LeCun](http://yann.lecun.com), [Amir Bar](https://www.amirbar.net/)<sup>†</sup>, [Saining Xie](https://www.sainingxie.com/)<sup>†</sup>

FAIR Meta, New York University, Princeton University  
<sup>*</sup>equal contribution, <sup>†</sup>equal advising

[<img src="https://img.shields.io/badge/arXiv-2504.01017-b31b1b.svg" height="22">](https://arxiv.org/abs/2504.01017)
[<img src="https://img.shields.io/badge/Project-Page-blue" height="22">](https://davidfan.io/webssl/)

<p align="center">
<img src="https://davidfan.io/webssl/assets/figures/fig1_simple_v2.png" width=75% height=75% 
class="center">
</p>


## Available Models
### Web-DINO Models
<table>
  <tr>
    <th colspan="1">model</th>
    <th colspan="1">patch size</th>
    <th colspan="1">resolution</th>
    <th colspan="1">data</th>
    <th colspan="2">download</th>
  </tr>
  <tr>
    <td>webssl-dino300m-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino300m-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino300m_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino1b-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino1b-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino1b_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino2b-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino2b-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino3b-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino3b-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino5b-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino5b-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino5b_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino7b-full8b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>8B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino7b-full8b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino7b-full8b-384</td>
    <td>14x14</td>
    <td>384x384</td>
    <td>8B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino7b-full8b-384">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_384.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino7b-full8b-518</td>
    <td>14x14</td>
    <td>518x518</td>
    <td>8B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino7b-full8b-518">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino7b_full8b_518.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino2b-light2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B light)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino2b-light2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_light2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino2b-heavy2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B heavy)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino2b-heavy2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino2b_heavy2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino3b-light2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B light)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino3b-light2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_light2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-dino3b-heavy2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B heavy)</td>
    <td><a href="https://huggingface.co/facebook/webssl-dino3b-heavy2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_dino3b_heavy2b_224.pth">PyTorch Weights</a></td>
  </tr>
</table>

  ### Web-MAE Models

  <table>
  <tr>
    <th colspan="1">model</th>
    <th colspan="1">patch size</th>
    <th colspan="1">resolution</th>
    <th colspan="1">data</th>
    <th colspan="2">download</th>
  </tr>
  <tr>
    <td>webssl-mae300m-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-mae300m-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_mae300m_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-mae700m-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-mae700m-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_mae700m_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-mae1b-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-mae1b-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_mae1b_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-mae2b-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-mae2b-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_mae2b_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-mae3b-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>2B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-mae3b-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_mae3b_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
  <tr>
    <td>webssl-mae7b-full2b-224</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>8B (MC-2B)</td>
    <td><a href="https://huggingface.co/facebook/webssl-mae7b-full2b-224">HuggingFace</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/webssl/webssl_mae7b_full2b_224.pth">PyTorch Weights</a></td>
  </tr>
</table>



## Usage

### Loading pretrained models

We provide two ways to use our models:

#### 1. Using HuggingFace Transformers

```python
from transformers import AutoImageProcessor, Dinov2Model

# Load a Web-DINO model
model_name = "facebook/webssl-dino1b-full2b-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = Dinov2Model.from_pretrained(model_name)

# Process an image
from PIL import Image
import requests

image = Image.open("path/to/image.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```

#### 2. Using PyTorch with original weights

```python
from dinov2.vision_transformer import vit_1b
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
model = vit_1b(
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

# Load weights
checkpoint_path = "path/to/downloaded/weights.pth"
state_dict = torch.load(checkpoint_path, map_location="cpu")
msg = model.load_state_dict(state_dict, strict=False)
model.eval()

# Process an image
image = Image.open("path/to/image.jpg")
x = transform(image).unsqueeze(0)
with torch.no_grad():
    features = model.forward_features(x)
    patch_features = features['x_norm_patchtokens']
```

See [demo.py](demo.py) for a complete example comparing HuggingFace and PyTorch implementations.



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

