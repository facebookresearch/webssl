# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from timm.models.vision_transformer import _create_vision_transformer

def webssl_mae300m_full2b_224():
    """
    Web-MAE ViT-300M
    ViT-L architecture
    """
    model = _create_vision_transformer('vit_large_patch16_224.mae', patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0)
    return model

def webssl_mae700m_full2b_224():
    """
    Web-MAE ViT-700M
    ViT-H architecture
    """
    model = _create_vision_transformer('vit_large_patch16_224.mae', patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0)
    return model

def webssl_mae1b_full2b_224():
    """
    Web-MAE ViT-1B
    DINOv2's "giant2" architecture / ViT-little g
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = _create_vision_transformer('vit_large_patch16_224.mae', patch_size=14, embed_dim=1536, depth=40, num_heads=24, mlp_ratio=4.0)
    return model

def webssl_mae2b_full2b_224():
    """Web-MAE ViT-2B (LLM-inspired scaling)"""
    model = _create_vision_transformer('vit_large_patch16_224.mae', patch_size=14, embed_dim=2688, depth=24, num_heads=21, mlp_ratio=4.0)
    return model

def webssl_mae3b_full2b_224():
    """Web-MAE ViT-3B (LLM-inspired scaling)"""
    model = _create_vision_transformer('vit_large_patch16_224.mae', patch_size=14, embed_dim=3072, depth=26, num_heads=24, mlp_ratio=4.0)
    return model

