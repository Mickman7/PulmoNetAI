"""
Ablation study across 4 input conditions:
  1) Vision only
  2) Text only
  3) Vision + Text
  4) Vision + Text + Labs (Proposed)

Design note: your MultimodalSystem architecture always uses text as the
cross-attention Query (image/labs are Key/Value), so there's no way to run
"vision only" or ablate a modality by zeroing its input on the ONE jointly
trained model -- it never learned to handle a zeroed vector meaningfully,
so that would not be a valid measurement.

Instead, each condition below is a separate lightweight classifier head,
trained on the SAME train/test split, reusing embeddings extracted ONCE from
your frozen SwinV2/ClinicalBERT encoders (they never change across
conditions since they're frozen) -- this keeps the 4-way comparison fast and
fair, since each head only has to learn the fusion/classification step.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from torch.utils.data import DataLoader, Dataset

from models.multimodal_system import MultimodalSystem

OUTPUT_DIR = "ablation_results"
EMBED_DIM = 512
NUM_EPOCHS = 5  # heads are small and train fast on cached embeddings


# ---------------------------------------------------------------------------
# 1. Extract and cache frozen-encoder embeddings ONCE (shared by all 4 heads)
# ---------------------------------------------------------------------------
def extract_embeddings(model, loader, device):
    """Runs the frozen encoders once and returns cached tensors for every head to reuse."""
    model.eval()
    img_seqs, text_pooled, labs, labels = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device)
            text_input = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            labs_tensor = torch.stack([batch["wbc"], batch["crp"]], dim=-1).float().to(device)

            img_features = model.image_encoder(images).last_hidden_state  # [B, num_patches, hidden]
            text_features = model.text_encoder(**text_input).pooler_output  # [B, hidden]

            img_seqs.append(img_features.cpu())
            text_pooled.append(text_features.cpu())
            labs.append(labs_tensor.cpu())
            labels.append(batch["labels"].cpu())

    return (
        torch.cat(img_seqs), torch.cat(text_pooled), torch.cat(labs), torch.cat(labels)
    )


class CachedEmbeddingDataset(Dataset):
    def __init__(self, img_seq, text_pooled, labs, labels):
        self.img_seq = img_seq
        self.text_pooled = text_pooled
        self.labs = labs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.img_seq[idx], self.text_pooled[idx], self.labs[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# 2. Four condition-specific heads
# ---------------------------------------------------------------------------
class VisionOnlyHead(nn.Module):
    """Mean-pools image patch tokens, no text/labs involved at all."""
    def __init__(self, img_hidden_size, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.img_proj = nn.Linear(img_hidden_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq, text_pooled=None, labs=None):
        img_vector = self.img_proj(img_seq)       # [B, num_patches, embed_dim]
        pooled = img_vector.mean(dim=1)            # mean-pool across patches
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class TextOnlyHead(nn.Module):
    """Text pooled embedding straight to a classifier, no image/labs involved."""
    def __init__(self, text_hidden_size, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Linear(text_hidden_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq=None, text_pooled=None, labs=None):
        text_vector = self.text_proj(text_pooled)
        text_vector = self.dropout(text_vector)
        return self.classifier(text_vector)


class VisionTextHead(nn.Module):
    """Your original architecture (before the labs branch was added):
    text as cross-attention Query, image patches as Key/Value."""
    def __init__(self, img_hidden_size, text_hidden_size, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.img_proj = nn.Linear(img_hidden_size, embed_dim)
        self.text_proj = nn.Linear(text_hidden_size, embed_dim)
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq, text_pooled, labs=None):
        img_vector = self.img_proj(img_seq)
        text_vector = self.text_proj(text_pooled)

        query = text_vector.unsqueeze(1)
        attn_out, _ = self.fusion(query, img_vector, img_vector)
        fused = self.norm(attn_out.squeeze(1) + text_vector)
        fused = self.dropout(fused)
        return self.classifier(fused)


class VisionTextLabsHead(nn.Module):
    """The 'Proposed' architecture -- matches your current MultimodalSystem
    exactly, just operating on cached embeddings instead of raw inputs."""
    def __init__(self, img_hidden_size, text_hidden_size, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.img_proj = nn.Linear(img_hidden_size, embed_dim)
        self.text_proj = nn.Linear(text_hidden_size, embed_dim)

        self.lab_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lab_proj = nn.Linear(32, embed_dim)

        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq, text_pooled, labs):
        img_vector = self.img_proj(img_seq)
        text_vector = self.text_proj(text_pooled)

        labs_formatted = labs.unsqueeze(1)  # [B, 1, 2]
        lab_features = self.lab_cnn(labs_formatted).squeeze(-1)  # [B, 32]
        lab_vector = self.lab_proj(lab_features)  # [B, embed_dim]
        lab_token = lab_vector.unsqueeze(1)  # [B, 1, embed_dim]

        kv_combined = torch.cat([img_vector, lab_token], dim=1)  # [B, num_patches+1, embed_dim]

        query = text_vector.unsqueeze(1)
        attn_out, attn_weights = self.fusion(query, kv_combined, kv_combined)
        fused = self.norm(attn_out.squeeze(1) + text_vector)
        fused = self.dropout(fused)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# 3. Generic train + evaluate loop for any head
# ---------------------------------------------------------------------------
def train_head(head, train_loader, device, num_epochs=NUM_EPOCHS, lr=1e-3):
    head.to(device)
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    head.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for img_seq, text_pooled, labs, labels in train_loader:
            img_seq, text_pooled, labs = img_seq.to(device), text_pooled.to(device), labs.to(device)
            labels = labels.to(device).float()

            optimiser.zero_grad()
            logits = head(img_seq, text_pooled, labs)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch + 1}/{num_epochs} -- loss: {total_loss / len(train_loader):.4f}")

    return head


def evaluate_head(head, test_loader, device):
    head.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for img_seq, text_pooled, labs, labels in test_loader:
            img_seq, text_pooled, labs = img_seq.to(device), text_pooled.to(device), labs.to(device)
            logits = head(img_seq, text_pooled, labs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    return all_probs, all_preds, all_labels


# ---------------------------------------------------------------------------
# 4. Run all 4 conditions and produce comparison outputs
# ---------------------------------------------------------------------------
def run_ablation_study(train_loader_raw, test_loader_raw, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load a MultimodalSystem purely to get the frozen encoders for embedding extraction
    print("Loading frozen encoders for embedding extraction...")
    encoder_model = MultimodalSystem(freeze_encoders=True).to(device)

    print("Extracting train embeddings (runs SwinV2/ClinicalBERT once)...")
    train_img, train_text, train_labs, train_labels = extract_embeddings(encoder_model, train_loader_raw, device)
    print("Extracting test embeddings...")
    test_img, test_text, test_labs, test_labels = extract_embeddings(encoder_model, test_loader_raw, device)

    train_ds = CachedEmbeddingDataset(train_img, train_text, train_labs, train_labels)
    test_ds = CachedEmbeddingDataset(test_img, test_text, test_labs, test_labels)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    img_hidden = encoder_model.image_encoder.config.hidden_size
    text_hidden = encoder_model.text_encoder.config.hidden_size

    conditions = {
        "Vision Only": VisionOnlyHead(img_hidden),
        "Text Only": TextOnlyHead(text_hidden),
        "Vision + Text": VisionTextHead(img_hidden, text_hidden),
        "Vision + Text + Labs (Proposed)": VisionTextLabsHead(img_hidden, text_hidden),
    }

    results = {}

    for name, head in conditions.items():
        print(f"\n{'=' * 60}\nTraining condition: {name}\n{'=' * 60}")
        head = train_head(head, train_loader, device)
        probs, preds, labels = evaluate_head(head, test_loader, device)

        # Classification report
        report = classification_report(labels, preds, target_names=["Normal", "Pneumonia"], digits=4)
        print(report)
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "and")
        with open(f"{OUTPUT_DIR}/{safe_name}_classification_report.txt", "w") as f:
            f.write(report)

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
        plt.title(f"Confusion Matrix -- {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{safe_name}_confusion_matrix.png", dpi=150)
        plt.close()

        # Store summary metrics for the combined comparison plots
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        results[name] = {
            "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "auc": roc_auc, "fpr": fpr, "tpr": tpr,
        }

    # --- Combined bar chart: Accuracy/Precision/Recall/F1 across all 4 conditions --
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k in ("accuracy", "precision", "recall", "f1")}
        for name, r in results.items()
    }).T

    ax = metrics_df.plot(kind="bar", figsize=(10, 6), rot=20)
    ax.set_title("Model Performance Across Input Conditions")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/combined_metrics_bar_chart.png", dpi=150)
    plt.close()
    print(f"\nCombined bar chart saved to {OUTPUT_DIR}/combined_metrics_bar_chart.png")

    # --- Combined ROC curve: one line per condition ------------------------------
    plt.figure(figsize=(7, 6))
    for name, r in results.items():
        plt.plot(r["fpr"], r["tpr"], label=f"{name} (AUC = {r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves -- All Conditions")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/combined_roc_curves.png", dpi=150)
    plt.close()
    print(f"Combined ROC curve saved to {OUTPUT_DIR}/combined_roc_curves.png")

    print(f"\nAll per-condition reports/confusion matrices saved to ./{OUTPUT_DIR}/")
    return results


if __name__ == "__main__":
    from datasets import load_from_disk

    # Same dataset path/format as model_training.py -- adjust here if yours differs
    DATASET_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed"

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Processed dataset not found at {DATASET_PATH}. Run your pipeline first.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading processed dataset...")
    dataset = load_from_disk(DATASET_PATH)

    split_ds = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]

    columns_to_load = ["pixel_values", "input_ids", "attention_mask", "labels", "wbc", "crp"]
    train_ds.set_format(type="torch", columns=columns_to_load)
    test_ds.set_format(type="torch", columns=columns_to_load)

    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    train_loader_raw = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    test_loader_raw = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    run_ablation_study(train_loader_raw, test_loader_raw, device)