"""MultimodalSystem — must match training-time architecture exactly."""

import torch.nn as nn
from transformers import AutoModel

IMAGE_MODEL_NAME = "microsoft/swinv2-tiny-patch4-window8-256"
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


class MultimodalSystem(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.3, freeze_encoders=True):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(IMAGE_MODEL_NAME)
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME)

        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.img_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)  # raw logits

    def forward(self, images, text_input, return_attention=False):
        img_features = self.image_encoder(images).last_hidden_state  # [B, num_patches, hidden]
        text_features = self.text_encoder(**text_input).pooler_output  # [B, hidden]

        img_vector = self.img_proj(img_features)
        text_vector = self.text_proj(text_features)

        query = text_vector.unsqueeze(1)
        attention_out, attn_weights = self.fusion(query, img_vector, img_vector)
        fused_vec = self.norm(attention_out.squeeze(1) + text_vector)
        fused_vec = self.dropout(fused_vec)
        logits = self.classifier(fused_vec)

        if return_attention:
            return logits, attn_weights
        return logits