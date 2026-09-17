import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from evaluation import run_evaluation
from huggingface_hub import hf_hub_download
from models.multimodal_system import MultimodalSystem
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

# 1. Argument Parsing
parser = argparse.ArgumentParser(
    description="PulmoNetAI Joint-Sequence Training Script"
)
parser.add_argument(
    "--lrs",
    nargs="+",
    type=float,
    default=[1e-3, 1e-5, 2e-5, 3e-5],
    help="List of learning rates to evaluate",
)
parser.add_argument(
    "--dropouts",
    nargs="+",
    type=float,
    default=[0.2, 0.3, 0.5],
    help="List of dropout probabilities to evaluate",
)
parser.add_argument(
    "--epochs", type=int, default=3, help="Number of training epochs per run"
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="Training batch size"
)
args = parser.parse_args()

# Device Selection
if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print(f"Using device: {device}")

# Dataset Path
DATASET_PATH = (
    "/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed"
)

if not os.path.exists(DATASET_PATH):
  raise FileNotFoundError(
      f"Processed dataset not found at {DATASET_PATH}. Run your pipeline first."
  )

print("Loading processed dataset...")
dataset = load_from_disk(DATASET_PATH)

# Split into train, validation, and test subsets
split_ds = dataset.train_test_split(test_size=0.2, seed=42)
train_val_ds = split_ds["train"]
test_ds = split_ds["test"]

sub_split = train_val_ds.train_test_split(test_size=0.2, seed=42)
train_ds = sub_split["train"]
val_ds = sub_split["test"]

# Enforce explicit formatting for PyTorch channels (Added 'vitals')
columns_to_load = [
    "pixel_values",
    "input_ids",
    "attention_mask",
    "labels",
    "wbc",
    "crp",
    "vitals",
]
train_ds.set_format(type="torch", columns=columns_to_load)
val_ds.set_format(type="torch", columns=columns_to_load)
test_ds.set_format(type="torch", columns=columns_to_load)

print(
    f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Test"
    f" samples: {len(test_ds)}"
)

train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
)

# 2. Model
base_model = MultimodalSystem(freeze_encoders=True).to(device)

# 3. Feature Extraction Cache Engine
os.makedirs("cached_embeddings", exist_ok=True)
base_model.eval()


def cache_split_features(loader, prefix):
  img_cache, text_cache, lab_cache, vitals_cache, label_cache = ([],[],[],[],[],)
  
  with torch.no_grad():
    for batch in loader:
      images = batch["pixel_values"].to(device)
      text_input = {
          "input_ids": batch["input_ids"].to(device),
          "attention_mask": batch["attention_mask"].to(device),
      }
      labs = torch.stack([batch["wbc"], batch["crp"]], dim=-1).float()
      vitals = batch["vitals"].float()  # (Batch, 24, 8)

      img_feat = base_model.image_encoder(images).last_hidden_state.cpu()
      text_feat = base_model.text_encoder(**text_input).pooler_output.cpu()

      img_cache.append(img_feat)
      text_cache.append(text_feat)
      lab_cache.append(labs)
      vitals_cache.append(vitals)
      label_cache.append(batch["labels"].cpu())

  torch.save(
      torch.cat(img_cache), f"cached_embeddings/{prefix}_img_features.pt"
  )
  torch.save(
      torch.cat(text_cache), f"cached_embeddings/{prefix}_text_features.pt"
  )
  torch.save(torch.cat(lab_cache), f"cached_embeddings/{prefix}_labs.pt")
  torch.save(torch.cat(vitals_cache), f"cached_embeddings/{prefix}_vitals.pt")
  torch.save(torch.cat(label_cache), f"cached_embeddings/{prefix}_labels.pt")


print("Extracting and caching features...")
cache_split_features(train_loader, "train")
cache_split_features(val_loader, "val")


class CachedDataset(Dataset):

  def __init__(self, prefix):
    self.img_feats = torch.load(f"cached_embeddings/{prefix}_img_features.pt")
    self.text_feats = torch.load(
        f"cached_embeddings/{prefix}_text_features.pt"
    )
    self.labs = torch.load(f"cached_embeddings/{prefix}_labs.pt")
    self.vitals = torch.load(f"cached_embeddings/{prefix}_vitals.pt")
    self.labels = torch.load(f"cached_embeddings/{prefix}_labels.pt")

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return (
        self.img_feats[idx].float(),
        self.text_feats[idx].float(),
        self.labs[idx].float(),
        self.vitals[idx].float(),
        self.labels[idx].float(),
    )


train_cached = CachedDataset("train")
val_cached = CachedDataset("val")

train_cached_loader = DataLoader(
    train_cached, batch_size=args.batch_size, shuffle=True, num_workers=0
)
val_cached_loader = DataLoader(
    val_cached, batch_size=args.batch_size, shuffle=False, num_workers=0
)

# 4. Hyperparameter Tuning Grid Search
best_val_loss = float("inf")
best_model_state = None

for lr in args.lrs:
  for dropout_val in args.dropouts:
    print(f"\n--- Tuning Profile: LR={lr} | Dropout={dropout_val} ---")

    model = MultimodalSystem(dropout=dropout_val, freeze_encoders=True).to(
        device
    )
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
      model.train()
      total_loss, correct_train, total_train = 0.0, 0, 0

      for img_feat, text_feat, labs, vitals, labels in train_cached_loader:
        optimiser.zero_grad()
        img_feat, text_feat, labs, vitals, labels = (
            img_feat.to(device),
            text_feat.to(device),
            labs.to(device),
            vitals.to(device),
            labels.to(device),
        )

        img_vector = model.img_proj(img_feat)
        text_vector = model.text_proj(text_feat)

        # Static Lab Branch
        labs_formatted = labs.unsqueeze(1)
        lab_features = model.lab_cnn(labs_formatted).squeeze(-1)
        lab_vector = model.lab_proj(lab_features)

        # Temporal Vitals Branch (1D CNN expecting [Batch, Channels, Length])
        vitals_formatted = vitals.transpose(1, 2)
        vitals_features = model.vitals_cnn(vitals_formatted).squeeze(-1)
        vitals_vector = model.vitals_proj(vitals_features)

        # Sequence tokens for Multihead Attention Fusion
        lab_token = lab_vector.unsqueeze(1)
        vitals_token = vitals_vector.unsqueeze(1)
        kv_combined = torch.cat(
            [img_vector, lab_token, vitals_token], dim=1
        )  # Merge image, labs, and time series

        query = text_vector.unsqueeze(1)
        attention_out, _ = model.fusion(query, kv_combined, kv_combined)

        fused_embeddings = model.norm(attention_out.squeeze(1) + text_vector)
        fused_embeddings = model.dropout(fused_embeddings)
        logits = model.classifier(fused_embeddings)

        loss = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

      train_loss = total_loss / len(train_cached_loader)
      train_acc = correct_train / total_train

      model.eval()
      val_loss_accum, correct_val, total_val = 0.0, 0, 0

      with torch.no_grad():
        for img_feat, text_feat, labs, vitals, labels in val_cached_loader:
          img_feat, text_feat, labs, vitals, labels = (
              img_feat.to(device),
              text_feat.to(device),
              labs.to(device),
              vitals.to(device),
              labels.to(device),
          )

          img_vector = model.img_proj(img_feat)
          text_vector = model.text_proj(text_feat)

          labs_formatted = labs.unsqueeze(1)
          lab_features = model.lab_cnn(labs_formatted).squeeze(-1)
          lab_vector = model.lab_proj(lab_features)

          vitals_formatted = vitals.transpose(1, 2)
          vitals_features = model.vitals_cnn(vitals_formatted).squeeze(-1)
          vitals_vector = model.vitals_proj(vitals_features)

          lab_token = lab_vector.unsqueeze(1)
          vitals_token = vitals_vector.unsqueeze(1)
          kv_combined = torch.cat([img_vector, lab_token, vitals_token], dim=1)

          query = text_vector.unsqueeze(1)
          attention_out, _ = model.fusion(query, kv_combined, kv_combined)

          fused_embeddings = model.norm(attention_out.squeeze(1) + text_vector)
          fused_embeddings = model.dropout(fused_embeddings)
          logits = model.classifier(fused_embeddings)

          loss = criterion(logits, labels)
          val_loss_accum += loss.item()

          preds = (torch.sigmoid(logits) > 0.5).float()
          correct_val += (preds == labels).sum().item()
          total_val += labels.size(0)

      val_loss = val_loss_accum / len(val_cached_loader)
      val_acc = correct_val / total_val

      print(
          f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f} |"
          f" Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc:"
          f" {val_acc:.2%}"
      )

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_model_state = model.state_dict()

final_model = MultimodalSystem(freeze_encoders=True).to(device)
final_model.load_state_dict(best_model_state)

# 5. File Serialisation
PATH = (
    "/Users/mickman/Documents/programs/PulmoNetAI/backend/models/multimodal_pneumonia_model.pth"
)
os.makedirs(os.path.dirname(PATH), exist_ok=True)

if os.path.exists(PATH):
  os.remove(PATH)

torch.save(final_model.state_dict(), PATH)
print(f"Optimised weights successfully deployed to production space: {PATH}")

run_evaluation(final_model, test_loader, device)