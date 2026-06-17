#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPImageProcessor
import llava.model.multimodal_encoder.mobileclip as mobileclip


class MobileCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.tune_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.input_image_size = int(vision_tower.split("_")[-1])

        # Delay load is disabled for now
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            model_cfg = mobileclip.load_model_config(self.vision_tower_name)
            self.cfg_only = model_cfg

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # Load model config
        model_cfg = mobileclip.load_model_config(self.vision_tower_name)

        # Override default image resolution
        model_cfg["image_cfg"]["image_size"] = self.input_image_size

        self.cfg_only = model_cfg

        # Build HF CLIPImageProcessor with MobileCLIP parameters
        self.image_processor = CLIPImageProcessor(crop_size={"height": model_cfg["image_cfg"]["image_size"],
                                                             "width": model_cfg["image_cfg"]["image_size"]},
                                                  image_mean=[0.0, 0.0, 0.0],
                                                  image_std=[1.0, 1.0, 1.0],
                                                  size={"shortest_edge": model_cfg["image_cfg"]["image_size"]})

        # Instantiate the image encoder
        self.vision_tower = mobileclip.MCi(model_name=model_cfg["image_cfg"]["model_name"],
                                           projection_dim=model_cfg["embed_dim"])

        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # Features from penultimate layer
        image_features = image_forward_outs["image_embeddings"]

        # Reshape 4D tensor to 3D
        B, C, H, W = image_features.shape
        image_features = image_features.reshape(B, C, H*W)
        image_features = image_features.transpose(1, 2)
        return image_features

    def forward(self, images):
        if self.tune_vision_tower:
            return self.forward_images(images)
        else:
            with torch.no_grad():
                return self.forward_images(images)

    def forward_images(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), return_image_embeddings=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), return_image_embeddings=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    def _vision_tower_tensor(self):
        for tensor in self.vision_tower.parameters():
            return tensor

        for module in self.vision_tower.modules():
            former_parameters = getattr(module, "_former_parameters", None)
            if former_parameters:
                for tensor in former_parameters.values():
                    return tensor

        for tensor in self.vision_tower.buffers():
            return tensor

        raise RuntimeError("MobileCLIP vision tower has no parameters or buffers.")

    @property
    def dtype(self):
        return self._vision_tower_tensor().dtype

    @property
    def device(self):
        return self._vision_tower_tensor().device

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config["image_cfg"]["embed_dim"]

    @property
    def num_patches_per_side(self):
        return self.config["image_cfg"]["image_size"] // self.config["image_cfg"]["patch_size"]

    @property
    def num_patches(self):
        return (self.config["image_cfg"]["image_size"] // self.config["image_cfg"]["patch_size"]) ** 2
