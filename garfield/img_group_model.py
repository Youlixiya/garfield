"""
Quick wrapper for Segment Anything Model
"""

from dataclasses import dataclass, field
from typing import Type, Union, Literal

import torch
import numpy as np

from PIL import Image

from nerfstudio.configs import base_config as cfg


@dataclass
class ImgGroupModelConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: ImgGroupModel)
    """target class to instantiate"""
    model_type: Literal["tap"] = "tap"
    """
    Currently supports:
     - "tap": Original TAP model
    """

    tap_model_type: str = ""
    tap_model_ckpt: str = ""
    tap_kwargs: dict = field(default_factory=lambda: {})
    "Arguments for SAM model (fb)."

    # # Settings used for the paper:
    # model_type="sam_fb",  
    # sam_model_type="vit_h",
    # sam_model_ckpt="models/sam_vit_h_4b8939.pth",
    # sam_kwargs={
    #     "points_per_side": 32,  # 32 in original
    #     "pred_iou_thresh": 0.90,
    #     "stability_score_thresh": 0.90,
    # },

    device: Union[torch.device, str] = ("cpu",)


class ImgGroupModel:
    """
    Wrapper for 2D image segmentation models (e.g. MaskFormer, SAM)
    Original paper uses SAM, but we can use any model that outputs masks.
    The code currently assumes that every image has at least one group/mask.
    """
    def __init__(self, config: ImgGroupModelConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.device = self.config.device = self.kwargs["device"]
        self.model = None

        # also, assert that model_type doesn't have a "/" in it! Will mess with h5df.
        assert "/" not in self.config.model_type, "model_type cannot have a '/' in it!"

    def __call__(self, img: np.ndarray):
        # takes in range 0-255... HxWx3
        # For using huggingface transformer's SAM model
        if self.config.model_type == "tap":
            # For using the original SAM model
            if self.model is None:
                from tokenize_anything import TapAutomaticMaskGenerator, model_registry
                model = model_registry[self.config.tap_model_type](checkpoint=self.config.tap_model_ckpt).cuda()
                model.text_decoder.reset_cache(max_batch_size=1024)
                model = model.to(device=self.config.device)
                self.model = TapAutomaticMaskGenerator(model=model,
                                                       **self.config.tap_kwargs
                )
                
            masks = self.model.generate(img)
            masks = [{'segmentation': m['segmentation'], 'caption': m['caption'], 'sem_token': m['sem_token'], 'bbox': m['bbox']} for m in masks] # already as bool
            masks = sorted(masks, key=lambda x: x['segmentation'].sum())
            return masks

