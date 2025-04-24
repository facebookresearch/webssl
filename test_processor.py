from transformers import AutoImageProcessor
from torchvision import transforms
import torch
from PIL import Image

def test_resolution(img_size):
    processor_hf = AutoImageProcessor.from_pretrained(f'facebook/webssl-dino7b-full8b-{img_size}')
    # processor_hf = AutoImageProcessor.from_pretrained('facebook/dinov2-giant', use_fast=True)
    processor_hf.crop_size = {
        'height': img_size,
        'width': img_size
    }
    processor_hf.size = {
        'height': img_size,
        'width': img_size
    }
    processor_hf.do_center_crop = False
    transform_pt = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC), # resize shortest side to img_size
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    im = Image.open('sample_images/bird.JPEG')
    out_im = processor_hf(im).pixel_values[0]
    out_pt = transform_pt(im)

    print(torch.abs(out_pt - out_im).sum())


test_resolution(224)
test_resolution(378)
test_resolution(518)