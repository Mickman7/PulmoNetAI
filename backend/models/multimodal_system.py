"""MultimodalSystem -- must match training-time architecture exactly.
Includes the labs branch (1D-CNN over WBC/CRP) added during training."""

import torch
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

        # --- LAB RESULTS BRANCH: 1D-CNN over [WBC, CRP] ---
        self.lab_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lab_proj = nn.Linear(32, embed_dim)

        self.img_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        # Fusion over the joint image-patch + lab-token sequence
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)  # raw logits

    def forward(self, images, text_input, labs_tensor, return_attention=False):
        img_features = self.image_encoder(images).last_hidden_state  # [B, num_patches, hidden]
        img_vector = self.img_proj(img_features)                      # [B, num_patches, embed_dim]

        text_features = self.text_encoder(**text_input).pooler_output  # [B, hidden]
        text_vector = self.text_proj(text_features)                    # [B, embed_dim]

        labs_formatted = labs_tensor.unsqueeze(1)                      # [B, 1, 2]
        lab_features = self.lab_cnn(labs_formatted).squeeze(-1)        # [B, 32]
        lab_vector = self.lab_proj(lab_features)                       # [B, embed_dim]

        lab_sequence_token = lab_vector.unsqueeze(1)                   # [B, 1, embed_dim]
        kv_combined = torch.cat([img_vector, lab_sequence_token], dim=1)  # [B, num_patches+1, embed_dim]

        query = text_vector.unsqueeze(1)                               # [B, 1, embed_dim]
        attention_out, attn_weights = self.fusion(query, kv_combined, kv_combined)

        fused_embeddings = self.norm(attention_out.squeeze(1) + text_vector)
        fused_embeddings = self.dropout(fused_embeddings)

        logits = self.classifier(fused_embeddings)

        if return_attention:
            return logits, attn_weights
        return logits