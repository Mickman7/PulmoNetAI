import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from sklearn.metrics import classification_report, confusion_matrix

from backend.evaluation import run_evaluation

# 1. Device Selection (Optimized for Mac M4 Hardware Acceleration)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Path where your dataset was successfully saved
DATASET_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Processed dataset not found at {DATASET_PATH}. Run your pipeline first.")

print("Loading processed dataset...")
dataset = load_from_disk(DATASET_PATH)

# Split into train and test sets
split_ds = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = split_ds['train']
test_ds = split_ds['test']

# Explicitly tell the dataset to return PyTorch Tensors
train_ds.set_format(type="torch", columns=["pixel_values", "input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["pixel_values", "input_ids", "attention_mask", "labels"])

print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

# DataLoaders (num_workers set to 0 on macOS to prevent MPS multi-process threading crashes)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

# 2. Model Definition
class MultimodalSystem(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.3, freeze_encoders=True):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        if freeze_encoders:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.img_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, images, text_input, return_attention=False):
        img_features = self.image_encoder(images).last_hidden_state          
        text_features = self.text_encoder(**text_input).pooler_output        

        img_vector = self.img_proj(img_features)      
        text_vector = self.text_proj(text_features)    

        query = text_vector.unsqueeze(1)                
        context = img_vector                            

        attention_out, attn_weights = self.fusion(query, context, context)
        fused_vec = self.norm(attention_out.squeeze(1) + text_vector)
        fused_vec = self.dropout(fused_vec)

        logits = self.classifier(fused_vec)

        if return_attention:
            return logits, attn_weights
        return logits

model = MultimodalSystem(freeze_encoders=True).to(device)

# 3. Create and Save Cached Embeddings
os.makedirs("cached_embeddings", exist_ok=True)

model.eval()
img_cache, text_cache, label_cache = [], [], []

print("Extracting and caching embeddings...")
with torch.no_grad():
    for batch in train_loader:
        images = batch['pixel_values'].to(device)
        text_input = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }

        img_feat = model.image_encoder(images).last_hidden_state.cpu()   
        text_feat = model.text_encoder(**text_input).pooler_output.cpu()  

        img_cache.append(img_feat)
        text_cache.append(text_feat)
        label_cache.append(batch['labels'].cpu())

torch.save(torch.cat(img_cache), "cached_embeddings/img_features.pt")
torch.save(torch.cat(text_cache), "cached_embeddings/text_features.pt")
torch.save(torch.cat(label_cache), "cached_embeddings/labels.pt")
print("Cached embeddings saved successfully.")

# 4. Custom Dataset for Pre-computed Features
class CachedDataset(Dataset):
    def __init__(self):
        self.img_feats = torch.load("cached_embeddings/img_features.pt")
        self.text_feats = torch.load("cached_embeddings/text_features.pt")
        self.labels = torch.load("cached_embeddings/labels.pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Using explicit float precision conversion for stable training
        return self.img_feats[idx].float(), self.text_feats[idx].float(), self.labels[idx].float()

cached_dataset = CachedDataset()
cached_loader = DataLoader(cached_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)

# 5. Training Infrastructure Setup
optimiser = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)

criterion = nn.BCEWithLogitsLoss()
num_epochs = 3

model.train()

# 6. Training Loop (Cleaned of CUDA-exclusive AMP dependencies)
for epoch in range(num_epochs):
    total_loss = 0.0
    print(f"Starting Epoch {epoch+1}/{num_epochs}")

    for img_feat, text_feat, labels in cached_loader:
        optimiser.zero_grad()

        img_feat = img_feat.to(device)
        text_feat = text_feat.to(device)
        labels = labels.to(device)

        # Standard precision pass optimized for MPS framework execution
        img_vector = model.img_proj(img_feat)
        text_vector = model.text_proj(text_feat)

        query = text_vector.unsqueeze(1)
        attention_out, _ = model.fusion(query, img_vector, img_vector)
        fused_vec = model.norm(attention_out.squeeze(1) + text_vector)
        fused_vec = model.dropout(fused_vec) if hasattr(model, 'dropout') else model.dropout(fused_vec)
        logits = model.classifier(fused_vec)

        loss = criterion(logits, labels)

        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Complete. Average Loss: {total_loss/len(cached_loader):.4f}")


# 7. Save model
PATH = "/Users/mickman/Documents/programs/PulmoNetAI/backend/models/multimodal_pneumonia_model.pth"

# Create the parent directory if it doesn't exist
parent_dir = os.path.dirname(PATH)
os.makedirs(parent_dir, exist_ok=True)

torch.save(model.state_dict(), PATH)
print(f"Model saved to {PATH}")

run_evaluation(model, test_loader, device)