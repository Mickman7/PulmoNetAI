import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

IMAGE_MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


"""MultimodalSystem -- cross-attention fusion architecture.

Reverted back to cross-attention (from a concat+MLP variant) because:
- Your dissertation's stated contribution is the cross-attention fusion
  mechanism itself, not just "combine everything and classify."
- Your agent pipeline (analysis_node, report template's Radiology Findings
  section, summarize_attention()) depends on real attention weights that
  only cross-attention produces -- concat+MLP has no attention to expose.
- This is the architecture your existing 95.17% test-accuracy run and full
  ablation study were validated against.

Both the labs (WBC/CRP) and vitals (24h time-series) branches are kept as
SEPARATE fusion tokens (not one replacing the other), matching what your
model_training.py already expects: Key/Value = image patches + lab token +
vitals token, Query = text.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

IMAGE_MODEL_NAME = "microsoft/swinv2-tiny-patch4-window8-256"
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

VITALS_CHANNELS = 8  # matches your existing cached vitals shape [B, 24, 8] -- confirm what these represent


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

        # --- Labs branch: 1D-CNN over [WBC, CRP] ---
        self.lab_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lab_proj = nn.Linear(32, embed_dim)

        # --- Vitals branch: 1D-CNN over 24h time-series, VITALS_CHANNELS channels ---
        self.vitals_cnn = nn.Sequential(
            nn.Conv1d(in_channels=VITALS_CHANNELS, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.vitals_proj = nn.Linear(32, embed_dim)

        # Fusion over the joint sequence: image patches + lab token + vitals token
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)  # raw logits, shape [B, 1]

    def forward(self, images, text_input, labs_tensor, vitals_tensor=None, return_attention=False):
        batch_size = images.size(0)

        img_features = self.image_encoder(images).last_hidden_state    # [B, num_patches, hidden]
        img_vector = self.img_proj(img_features)                        # [B, num_patches, embed_dim]

        text_features = self.text_encoder(**text_input).pooler_output   # [B, hidden]
        text_vector = self.text_proj(text_features)                     # [B, embed_dim]

        # Labs: [B, 2] -> [B, 1, 2] for Conv1d
        labs_formatted = labs_tensor.unsqueeze(1) if labs_tensor.dim() == 2 else labs_tensor
        lab_features = self.lab_cnn(labs_formatted).squeeze(-1)         # [B, 32]
        lab_vector = self.lab_proj(lab_features)                        # [B, embed_dim]
        lab_token = lab_vector.unsqueeze(1)                             # [B, 1, embed_dim]

        # Vitals: optional. [B, 24, VITALS_CHANNELS] -> transpose -> [B, VITALS_CHANNELS, 24] for Conv1d.
        # None falls back to a zero token (non-informative -- doesn't push the
        # prediction toward either class), same pattern you already used in
        # your concat-model version, just sized to embed_dim here instead.
        if vitals_tensor is not None:
            vitals_formatted = vitals_tensor.transpose(1, 2)             # [B, VITALS_CHANNELS, 24]
            vitals_features = self.vitals_cnn(vitals_formatted).squeeze(-1)  # [B, 32]
            vitals_vector = self.vitals_proj(vitals_features)            # [B, embed_dim]
        else:
            vitals_vector = torch.zeros((batch_size, self.embed_dim), device=images.device, dtype=img_vector.dtype)
        vitals_token = vitals_vector.unsqueeze(1)                        # [B, 1, embed_dim]

        # Key/Value pool: image patches + lab token + vitals token
        kv_combined = torch.cat([img_vector, lab_token, vitals_token], dim=1)  # [B, num_patches+2, embed_dim]

        query = text_vector.unsqueeze(1)                                 # [B, 1, embed_dim]
        attention_out, attn_weights = self.fusion(query, kv_combined, kv_combined)

        fused_embeddings = self.norm(attention_out.squeeze(1) + text_vector)
        fused_embeddings = self.dropout(fused_embeddings)

        logits = self.classifier(fused_embeddings)  # [B, 1] -- NOT squeezed, matches labels shape [B, 1] used elsewhere

        if return_attention:
            return logits, attn_weights
        return logits