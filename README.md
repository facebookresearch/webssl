# Web-SSL: Scaling Language-Free Visual Representation Learning
Official inference code for the Web-SSL family of models.

[David Fan](https://davidfan.io), [Shengbang Tong](https://tsb0601.github.io/), [Jiachen Zhu](https://jiachenzhu.github.io), [Koustuv Sinha](https://koustuvsinha.com/), [Zhuang Liu](https://liuzhuang13.github.io), [Xinlei Chen](https://xinleic.xyz/), [Michael Rabbat](https://scholar.google.com/citations?user=cMPKe9UAAAAJ), [Nicolas Ballas](https://scholar.google.com/citations?user=euUV4iUAAAAJ), [Yann LeCun](http://yann.lecun.com), [Amir Bar](https://www.amirbar.net/), [Saining Xie](https://www.sainingxie.com/)

Meta FAIR, NYU, Princeton \
[[`arXiv`](https://arxiv.org/abs/2504.01017)][[`project page`](https://davidfan.io/webssl/)]

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