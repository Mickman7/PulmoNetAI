import os
import argparse
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

from evaluation import run_evaluation

# 1. Argument Parsing
parser = argparse.ArgumentParser(description="PulmoNetAI Tri-Modal Training Script")
parser.add_argument("--lrs", nargs="+", type=float, default=[1e-5, 3e-5], help="List of learning rates to evaluate")
parser.add_argument("--dropouts", nargs="+", type=float, default=[0.3, 0.5], help="List of dropout probabilities to evaluate")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per run")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
args = parser.parse_args()

# Device Selection (Optimized for Mac M4 Hardware Acceleration)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Dataset Path
DATASET_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Processed dataset not found at {DATASET_PATH}. Run your pipeline first.")

print("Loading processed dataset...")
dataset = load_from_disk(DATASET_PATH)

# Split into train, validation, and test subsets
split_ds = dataset.train_test_split(test_size=0.2, seed=42)
train_val_ds = split_ds['train']
test_ds = split_ds['test']

sub_split = train_val_ds.train_test_split(test_size=0.2, seed=42)
train_ds = sub_split['train']
val_ds = sub_split['test']

# Enforce explicit formatting for PyTorch channels
columns_to_load = ["pixel_values", "input_ids", "attention_mask", "labels", "wbc", "crp"]
train_ds.set_format(type="torch", columns=columns_to_load)
val_ds.set_format(type="torch", columns=columns_to_load)
test_ds.set_format(type="torch", columns=columns_to_load)

print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Test samples: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)


# 2. Tri-Modal Architecture Definition
class MultimodalSystem(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.3, freeze_encoders=True):
        super().__init__()
        # Encoders
        self.image_encoder = AutoModel.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        if freeze_encoders:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # --- LAB RESULTS BRANCH: 1D-CNN ---
        # Input tensor shape: [Batch, Channels=1, Sequence_Length=2] (representing [WBC, CRP])
        self.lab_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)  # Collapses sequence down to [Batch, 32, 1]
        )
        self.lab_proj = nn.Linear(32, embed_dim)

        # Projections for Vision and Text
        self.img_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        # Multi-Modal Fusion Layers
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Classification layer processes the concatenated embeddings from all 3 modalities
        self.classifier = nn.Linear(embed_dim * 2, 1)

    def forward(self, images, text_input, labs_tensor, return_attention=False):
        # 1. Process Images via Swin
        img_features = self.image_encoder(images).last_hidden_state  # [Batch, Patches, Hidden]
        img_vector = self.img_proj(img_features)                     # [Batch, Patches, Embed_Dim]

        # 2. Process Text via ClinicalBERT
        text_features = self.text_encoder(**text_input).pooler_output  # [Batch, Hidden]
        text_vector = self.text_proj(text_features)                    # [Batch, Embed_Dim]

        # 3. Process Physiological Lab Data via 1D-CNN
        # Reshape labs from [Batch, 2] to [Batch, Channels=1, Sequence=2]
        labs_formatted = labs_tensor.unsqueeze(1)
        lab_features = self.lab_cnn(labs_formatted).squeeze(-1)         # [Batch, 32]
        lab_vector = self.lab_proj(lab_features)                        # [Batch, Embed_Dim]

        # 4. Multi-Modal Attention Cross-Fusion (Image and Text)
        query = text_vector.unsqueeze(1)                
        attention_out, attn_weights = self.fusion(query, img_vector, img_vector)
        fused_text_img = self.norm(attention_out.squeeze(1) + text_vector)
        fused_text_img = self.dropout(fused_text_img)

        # 5. Final Concat Fusion (Combining Attended Text/Image with the Lab 1D-CNN Vector)
        final_flat_vector = torch.cat([fused_text_img, lab_vector], dim=-1) # [Batch, Embed_Dim * 2]
        logits = self.classifier(final_flat_vector)

        if return_attention:
            return logits, attn_weights
        return logits

# Initialize base prototype template for feature extraction caching
base_model = MultimodalSystem(freeze_encoders=True).to(device)


# 3. Create and Save Cached Embeddings (Now including raw Lab sequences)
os.makedirs("cached_embeddings", exist_ok=True)
base_model.eval()

def cache_split_features(loader, prefix):
    img_cache, text_cache, lab_cache, label_cache = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['pixel_values'].to(device)
            text_input = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            # Stack WBC and CRP together into a single sequence vector row
            labs = torch.stack([batch["wbc"], batch["crp"]], dim=-1).float()

            img_feat = base_model.image_encoder(images).last_hidden_state.cpu()   
            text_feat = base_model.text_encoder(**text_input).pooler_output.cpu()  
            
            img_cache.append(img_feat)
            text_cache.append(text_feat)
            lab_cache.append(labs)
            label_cache.append(batch['labels'].cpu())

    torch.save(torch.cat(img_cache), f"cached_embeddings/{prefix}_img_features.pt")
    torch.save(torch.cat(text_cache), f"cached_embeddings/{prefix}_text_features.pt")
    torch.save(torch.cat(lab_cache), f"cached_embeddings/{prefix}_labs.pt")
    torch.save(torch.cat(label_cache), f"cached_embeddings/{prefix}_labels.pt")

print("Extracting and caching tri-modal train/validation split features...")
cache_split_features(train_loader, "train")
cache_split_features(val_loader, "val")
print("Cached split embeddings saved successfully.")


# 4. Custom Dataset for Pre-computed Features
class CachedDataset(Dataset):
    def __init__(self, prefix):
        self.img_feats = torch.load(f"cached_embeddings/{prefix}_img_features.pt")
        self.text_feats = torch.load(f"cached_embeddings/{prefix}_text_features.pt")
        self.labs = torch.load(f"cached_embeddings/{prefix}_labs.pt")
        self.labels = torch.load(f"cached_embeddings/{prefix}_labels.pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.img_feats[idx].float(), self.text_feats[idx].float(), self.labs[idx].float(), self.labels[idx].float()

train_cached = CachedDataset("train")
val_cached = CachedDataset("val")

train_cached_loader = DataLoader(train_cached, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_cached_loader = DataLoader(val_cached, batch_size=args.batch_size, shuffle=False, num_workers=0)


# 5. Hyperparameter Tuning Grid
best_val_loss = float("inf")
best_model_state = None

for lr in args.lrs:
    for dropout_val in args.dropouts:
        print(f"\n--- Launching Tuning Profile: LR={lr} | Dropout={dropout_val} ---")
        
        model = MultimodalSystem(dropout=dropout_val, freeze_encoders=True).to(device)
        optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(args.epochs):
            # Training Mode
            model.train()
            total_loss = 0.0
            correct_train = 0
            total_train = 0

            for img_feat, text_feat, labs, labels in train_cached_loader:
                optimiser.zero_grad()
                img_feat, text_feat, labs, labels = img_feat.to(device), text_feat.to(device), labs.to(device), labels.to(device)

                # Process cache metrics through projections and the 1D-CNN
                img_vector = model.img_proj(img_feat)
                text_vector = model.text_proj(text_feat)
                
                labs_formatted = labs.unsqueeze(1)
                lab_features = model.lab_cnn(labs_formatted).squeeze(-1)
                lab_vector = model.lab_proj(lab_features)

                query = text_vector.unsqueeze(1)
                attention_out, _ = model.fusion(query, img_vector, img_vector)
                fused_text_img = model.norm(attention_out.squeeze(1) + text_vector)
                fused_text_img = model.dropout(fused_text_img)

                final_flat_vector = torch.cat([fused_text_img, lab_vector], dim=-1)
                logits = model.classifier(final_flat_vector)

                loss = criterion(logits, labels)
                loss.backward()
                optimiser.step()

                total_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            train_loss = total_loss / len(train_cached_loader)
            train_acc = correct_train / total_train

            # Validation Mode
            model.eval()
            val_loss_accum = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for img_feat, text_feat, labs, labels in val_cached_loader:
                    img_feat, text_feat, labs, labels = img_feat.to(device), text_feat.to(device), labs.to(device), labels.to(device)

                    img_vector = model.img_proj(img_feat)
                    text_vector = model.text_proj(text_feat)
                    
                    labs_formatted = labs.unsqueeze(1)
                    lab_features = model.lab_cnn(labs_formatted).squeeze(-1)
                    lab_vector = model.lab_proj(lab_features)

                    query = text_vector.unsqueeze(1)
                    attention_out, _ = model.fusion(query, img_vector, img_vector)
                    fused_text_img = model.norm(attention_out.squeeze(1) + text_vector)
                    fused_text_img = model.dropout(fused_text_img)

                    final_flat_vector = torch.cat([fused_text_img, lab_vector], dim=-1)
                    logits = model.classifier(final_flat_vector)

                    loss = criterion(logits, labels)
                    val_loss_accum += loss.item()

                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            val_loss = val_loss_accum / len(val_cached_loader)
            val_acc = correct_val / total_val

            print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

# Load weights of the best performing variant
final_model = MultimodalSystem(freeze_encoders=True).to(device)
final_model.load_state_dict(best_model_state)


# 6. Model File Save & Overwrite
PATH = "/Users/mickman/Documents/programs/PulmoNetAI/backend/models/multimodal_pneumonia_model.pth"
parent_dir = os.path.dirname(PATH)
os.makedirs(parent_dir, exist_ok=True)

if os.path.exists(PATH):
    print(f"Stale model variant detected at target path. Overwriting {PATH}...")
    os.remove(PATH)

torch.save(final_model.state_dict(), PATH)
print(f"Optimised tri-modal model weights written to file system space: {PATH}")

# Note: Update your run_evaluation script parameters if you change the test execution dimensions.
run_evaluation(final_model, test_loader, device)