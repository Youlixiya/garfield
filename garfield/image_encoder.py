from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy
from nerfstudio.viewer.viewer_elements import ViewerText
import torch
import torchvision
from torch import nn

try:
    import open_clip
    import alpha_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

@dataclass
class AlphaCLIPNetworkConfig:
    # _target: Type = field(default_factory=lambda: AlphaCLIPNetwork)
    clip_model_type: str = "ViT-B/16"
    clip_model_pretrained: str = "ckpts/clip_b16_grit+mim_fultune_4xe.pth"
    clip_n_dims: int = 512
    # negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    negatives: Tuple[str] = ("background")
    positives: Tuple[str] = ("",)

class AlphaCLIPNetwork(nn.Module):
    def __init__(self, config: AlphaCLIPNetworkConfig):
        super().__init__()
        self.config = config
        model, self.process = alpha_clip.load(config.clip_model_type, alpha_vision_ckpt_pth=config.clip_model_pretrained, device='cuda') 
        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(0.5, 0.26)
            ])
        model.eval()
        self.tokenizer = alpha_clip.tokenize
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        self.positive_input = ViewerText("LERF Positives", "", cb_hook=self.gui_cb)
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
        self.positive_input = ViewerText("LERF Positives", "", cb_hook=self.gui_cb)
        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "alphaclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    # def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
    #     phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
    #     embed /= embed.norm(dim=-1, keepdim=True)
    #     p = phrases_embeds.to(embed.dtype)  # phrases x 512
    #     output = torch.mm(embed, p.T)  # rays x phrases
    #     positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
    #     negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
    #     repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

    #     sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
    #     softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
    #     best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
    #     return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]
    
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        embed /= embed.norm(dim=-1, keepdim=True)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        sims = torch.cat([positive_vals, negative_vals], dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        return softmax
    
    def encode_image(self, input, mask):
        alpha_input = self.mask_transform(mask)
        alpha_input = alpha_input.half().to('cuda').unsqueeze(dim=0)
        processed_input = self.process(input).to('cuda').unsqueeze(0).half()
        with torch.no_grad():
            image_embedding = self.model.visual(processed_input, alpha_input)
            image_embedding = image_embedding.norm(dim=-1, keepdim=True)
        return image_embedding

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        self.positive_input = ViewerText("LERF Positives", "", cb_hook=self.gui_cb)
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        with torch.no_grad():
            processed_input = self.process(input).half().cuda().unsqueeze(0)
            return self.model.encode_image(processed_input)