import torch
import torch.nn as nn
from transformers import AutoModel

IMAGE_MODEL_NAME = "microsoft/swinv2-tiny-patch4-window8-256"
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
VITALS_CHANNELS = 8


class MultimodalSystem(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.3, freeze_encoders=True):
        super().__init__()
        self.embed_dim = embed_dim

        self.image_encoder = AutoModel.from_pretrained(IMAGE_MODEL_NAME)
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME)

        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.img_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        self.lab_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lab_proj = nn.Linear(32, embed_dim)

        self.vitals_cnn = nn.Sequential(
            nn.Conv1d(in_channels=VITALS_CHANNELS, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.vitals_proj = nn.Linear(32, embed_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, images, text_input, labs_tensor, vitals_tensor=None, return_attention=False):
        batch_size = images.size(0)

        img_features = self.image_encoder(images).last_hidden_state
        img_vector = self.img_proj(img_features)

        text_features = self.text_encoder(**text_input).pooler_output
        text_vector = self.text_proj(text_features).unsqueeze(1)

        labs_formatted = labs_tensor.unsqueeze(1) if labs_tensor.dim() == 2 else labs_tensor
        lab_features = self.lab_cnn(labs_formatted).squeeze(-1)
        lab_token = self.lab_proj(lab_features).unsqueeze(1)

        if vitals_tensor is not None:
            vitals_formatted = vitals_tensor.transpose(1, 2)
            vitals_features = self.vitals_cnn(vitals_formatted).squeeze(-1)
            vitals_vector = self.vitals_proj(vitals_features)
        else:
            vitals_vector = torch.zeros((batch_size, self.embed_dim), device=images.device, dtype=img_vector.dtype)
        vitals_token = vitals_vector.unsqueeze(1)

        joint_sequence = torch.cat([img_vector, text_vector, lab_token, vitals_token], dim=1)

        attn_out, attn_weights = self.self_attention(
            joint_sequence, joint_sequence, joint_sequence, need_weights=True
        )

        pooled_sequence = attn_out.mean(dim=1)
        fused_embeddings = self.norm(pooled_sequence)
        fused_embeddings = self.dropout(fused_embeddings)

        logits = self.classifier(fused_embeddings)

        if return_attention:
            return logits, attn_weights
        return logits