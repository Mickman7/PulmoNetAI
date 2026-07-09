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

        # Multi-Modal Fusion Layers (Processes the joint 65-token sequence pool)
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Classification layer handles the single, unified 512-dim embedding
        self.classifier = nn.Linear(embed_dim, 1)

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

        # 4. Construct the Joint Sequence Context Pool (K and V)
        # Add sequence dimension to labs: [Batch, 512] -> [Batch, 1, 512]
        lab_sequence_token = lab_vector.unsqueeze(1)

        # Concatenate image patches and lab token along the sequence dimension (dim 1)
        # Resulting shape: [Batch, 65, 512]
        kv_combined = torch.cat([img_vector, lab_sequence_token], dim=1)

        # 5. Cross-Attention Multihead Execution
        query = text_vector.unsqueeze(1)                
        attention_out, attn_weights = self.fusion(query, kv_combined, kv_combined)
        
        # Residual Connection
        fused_embeddings = self.norm(attention_out.squeeze(1) + text_vector)
        fused_embeddings = self.dropout(fused_embeddings)

        # 6. Final Class Prediction
        logits = self.classifier(fused_embeddings)

        if return_attention:
            return logits, attn_weights
        return logits