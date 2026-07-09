import torch
import torch.nn as nn
from transformers import AutoModel

IMAGE_MODEL_NAME = "microsoft/swinv2-tiny-patch4-window8-256"
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

class MultimodalSystem(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.3, freeze_encoders=True):
        super().__init__()
        # Encoders
        self.image_encoder = AutoModel.from_pretrained(IMAGE_MODEL_NAME)
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME)

        if freeze_encoders:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # --- LAB RESULTS BRANCH: 1D-CNN ---
        self.lab_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        self.lab_proj = nn.Linear(32, embed_dim)

        # Projections
        self.img_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        # Multi-Modal Fusion Layers
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Classification layer processes the combined text/image and lab outputs (512 * 2 = 1024)
        self.classifier = nn.Linear(embed_dim * 2, 1)

    def forward(self, images, text_input, labs_tensor, return_attention=False):
        # 1. Process Images
        img_features = self.image_encoder(images).last_hidden_state  
        img_vector = self.img_proj(img_features)                     

        # 2. Process Text
        text_features = self.text_encoder(**text_input).pooler_output  
        text_vector = self.text_proj(text_features)                    

        # 3. Process Lab Data via 1D-CNN
        labs_formatted = labs_tensor.unsqueeze(1)
        lab_features = self.lab_cnn(labs_formatted).squeeze(-1)         
        lab_vector = self.lab_proj(lab_features)                        

        # 4. Multi-Modal Cross-Attention (Image and Text)
        query = text_vector.unsqueeze(1)                
        attention_out, attn_weights = self.fusion(query, img_vector, img_vector)
        fused_text_img = self.norm(attention_out.squeeze(1) + text_vector)
        fused_text_img = self.dropout(fused_text_img)

        # 5. Concat Fusion with Lab Vector
        final_flat_vector = torch.cat([fused_text_img, lab_vector], dim=-1) 
        logits = self.classifier(final_flat_vector)

        if return_attention:
            return logits, attn_weights
        return logits