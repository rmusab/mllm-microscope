import torch
from torch import nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class VisualToGPTMapping(nn.Module):
    def __init__(self, visual_emb_dim, gpt_emb_dim, num_gpt_embs, num_heads):
        super(VisualToGPTMapping, self).__init__()
        self.transformer_layer = TransformerEncoderLayer(d_model=visual_emb_dim, nhead=num_heads, batch_first=True, norm_first=False)
        self.linear = Linear(visual_emb_dim, gpt_emb_dim)
        self.n_embeddings = num_gpt_embs
        self.embedding_dim = gpt_emb_dim
    def forward(self, visual_embs):
        out = self.transformer_layer(visual_embs)
        out = self.linear(out).view(-1, self.n_embeddings, self.embedding_dim)
        return out


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.select_feature = 'patch'

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
