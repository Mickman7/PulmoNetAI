import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

from models.multimodal_system import MultimodalSystem

OUTPUT_DIR = "ablation_results"
EMBED_DIM = 512
NUM_EPOCHS = 15


# ---------------------------------------------------------------------------
# 1. Early Stopping Utility
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# 2. Extract and cache frozen-encoder embeddings
# ---------------------------------------------------------------------------
def extract_embeddings(model, loader, device):
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

            img_features = model.image_encoder(images).last_hidden_state
            text_features = model.text_encoder(**text_input).pooler_output

            img_seqs.append(img_features.cpu())
            text_pooled.append(text_features.cpu())
            labs.append(labs_tensor.cpu())
            labels.append(batch["labels"].cpu())

    return (
        torch.cat(img_seqs),
        torch.cat(text_pooled),
        torch.cat(labs),
        torch.cat(labels),
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
# 3. Model Heads with Gated Multimodal Fusion
# ---------------------------------------------------------------------------
class VisionOnlyHead(nn.Module):
    def __init__(self, img_hidden_size, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.img_proj = nn.Linear(img_hidden_size, embed_dim)
        self.img_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq, text_pooled=None, labs=None, **kwargs):
        img_vector = self.img_norm(self.img_proj(img_seq))
        pooled = img_vector.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class TextOnlyHead(nn.Module):
    def __init__(self, text_hidden_size, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Linear(text_hidden_size, embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq=None, text_pooled=None, labs=None, **kwargs):
        text_vector = self.text_norm(self.text_proj(text_pooled))
        text_vector = self.dropout(text_vector)
        return self.classifier(text_vector)


class VisionTextHead(nn.Module):
    def __init__(self, img_hidden_size, text_hidden_size, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.img_proj = nn.Linear(img_hidden_size, embed_dim)
        self.text_proj = nn.Linear(text_hidden_size, embed_dim)

        self.img_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)

        # Gated fusion mechanism replacing raw addition
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.project = nn.Linear(embed_dim * 2, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq, text_pooled, labs=None, **kwargs):
        img_vector = self.img_norm(self.img_proj(img_seq))
        text_vector = self.text_norm(self.text_proj(text_pooled))

        if self.training and torch.rand(1).item() < 0.15:
            img_vector = torch.zeros_like(img_vector)

        img_token = img_vector.mean(dim=1)
        
        concatenated = torch.cat([img_token, text_vector], dim=1)
        gates = self.gate_net(concatenated)
        projected = self.project(concatenated)
        
        fused = self.norm(gates * projected + text_vector)
        fused = self.dropout(fused)
        return self.classifier(fused)


class VisionTextLabsHead(nn.Module):
    """Sequence-based Cross-Attention Fusion with Gated Modality Integration."""
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

        self.img_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.lab_norm = nn.LayerNorm(embed_dim)

        self.modality_dropout = nn.Dropout(p=0.15)

        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # Gated fusion to balance attention outputs and text signal
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.project = nn.Linear(embed_dim * 2, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq, text_pooled, labs, return_visuals=False):
        img_vector = self.img_norm(self.img_proj(img_seq))
        text_vector = self.text_norm(self.text_proj(text_pooled))

        labs_formatted = labs.unsqueeze(1)
        lab_features = self.lab_cnn(labs_formatted).squeeze(-1)
        lab_vector = self.lab_norm(self.lab_proj(lab_features))

        img_token = img_vector.mean(dim=1, keepdim=True)
        lab_token = lab_vector.unsqueeze(1)

        if self.training:
            img_token = self.modality_dropout(img_token)

        kv_sequence = torch.cat([img_token, lab_token], dim=1)
        query = text_vector.unsqueeze(1)

        attn_out, attn_weights = self.fusion(
            query,
            kv_sequence,
            kv_sequence,
            need_weights=True,
            average_attn_weights=True,
        )

        attn_vector = attn_out.squeeze(1)
        concatenated = torch.cat([attn_vector, text_vector], dim=1)
        gates = self.gate_net(concatenated)
        projected = self.project(concatenated)

        fused = self.norm(gates * projected + text_vector)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        if return_visuals:
            return logits, attn_weights.squeeze(1)

        return logits


# ---------------------------------------------------------------------------
# 4. Training and Evaluation Loops
# ---------------------------------------------------------------------------
def _compute_val_loss(head, val_loader, device, criterion):
    head.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img_seq, text_pooled, labs, labels in val_loader:
            img_seq, text_pooled, labs = img_seq.to(device), text_pooled.to(device), labs.to(device)
            labels = labels.to(device).float()
            logits = head(img_seq, text_pooled, labs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
    head.train()
    return total_loss / len(val_loader)


def train_head(head, train_loader, val_loader, device, num_epochs=NUM_EPOCHS, lr=1e-3):
    head.to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    stopper = EarlyStopping(patience=3, min_delta=0.001)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

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

        train_loss = total_loss / len(train_loader)
        val_loss = _compute_val_loss(head, val_loader, device, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            marker = "  <- best so far"

        print(f"    Epoch {epoch + 1}/{num_epochs} -- train loss: {train_loss:.4f} | "
              f"val loss: {val_loss:.4f}{marker}")

        stopper(val_loss)
        if stopper.early_stop:
            print(f"    Early stopping triggered at epoch {epoch + 1}")
            break

    if best_state is not None:
        head.load_state_dict(best_state)
        print(f"    Restored best checkpoint (val loss: {best_val_loss:.4f})")

    head.eval()
    return head, history


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


def plot_training_curves(all_histories: dict):
    fig, axes = plt.subplots(1, len(all_histories), figsize=(5 * len(all_histories), 4), sharey=True)
    if len(all_histories) == 1:
        axes = [axes]

    for ax, (name, history) in zip(axes, all_histories.items()):
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="Train Loss")
        ax.plot(epochs, history["val_loss"], label="Val Loss")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend()

    axes[0].set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves_all_conditions.png", dpi=150)
    plt.close()
    print(f"Training curves saved to {OUTPUT_DIR}/training_curves_all_conditions.png")


# ---------------------------------------------------------------------------
# 5. Extract and Plot Sequence Cross-Attention Weights
# ---------------------------------------------------------------------------
def visualize_modality_dynamics(proposed_head, test_loader, device):
    proposed_head.eval()
    proposed_head.to(device)

    all_attn_weights = []
    all_labels = []

    with torch.no_grad():
        for img_seq, text_pooled, labs, labels in test_loader:
            img_seq = img_seq.to(device)
            text_pooled = text_pooled.to(device)
            labs = labs.to(device)

            _, attn_weights = proposed_head(img_seq, text_pooled, labs, return_visuals=True)

            all_attn_weights.append(attn_weights.cpu().numpy())
            all_labels.extend(labels.numpy().flatten())

    all_attn_weights = np.vstack(all_attn_weights)
    all_labels = np.array(all_labels)

    vision_weights = all_attn_weights[:, 0]
    lab_weights = all_attn_weights[:, 1]

    print("\n" + "=" * 60)
    print("MODALITY SELF-ATTENTION ANALYSIS (Test Set)")
    print("=" * 60)
    print(f"Mean Text-to-Vision Attention: {np.mean(vision_weights) * 100:.2f}% (std: {np.std(vision_weights):.4f})")
    print(f"Mean Text-to-Lab Attention:    {np.mean(lab_weights) * 100:.2f}% (std: {np.std(lab_weights):.4f})")

    norm_mask = all_labels == 0
    pneu_mask = all_labels == 1
    print(f"  - Normal Cases -> Vision: {np.mean(vision_weights[norm_mask]) * 100:.2f}% | Labs: {np.mean(lab_weights[norm_mask]) * 100:.2f}%")
    print(f"  - Pneumonia Cases -> Vision: {np.mean(vision_weights[pneu_mask]) * 100:.2f}% | Labs: {np.mean(lab_weights[pneu_mask]) * 100:.2f}%")

    plt.figure(figsize=(8, 5))
    sns.kdeplot(vision_weights, label="Vision Token Attention Weight", fill=True, alpha=0.4, color="royalblue")
    sns.kdeplot(lab_weights, label="Lab Token Attention Weight", fill=True, alpha=0.4, color="crimson")
    plt.title("Sequence Cross-Attention Weight Distribution")
    plt.xlabel("Attention Weight Score (0.0 to 1.0)")
    plt.ylabel("Density")
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sequence_attention_distribution.png", dpi=150)
    plt.close()

    df_attn = pd.DataFrame({
        "Vision Weight": vision_weights,
        "Lab Weight": lab_weights,
        "Diagnosis": ["Normal" if l == 0 else "Pneumonia" for l in all_labels]
    })

    plt.figure(figsize=(7, 5))
    sns.boxplot(
        x="Diagnosis",
        y="Lab Weight",
        hue="Diagnosis",
        data=df_attn,
        palette=["mediumseagreen", "indianred"],
        legend=False,
    )
    plt.title("Laboratory Token Cross-Attention Score by Diagnosis Class")
    plt.ylabel("Learned Cross-Attention Weight")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lab_attention_by_class.png", dpi=150)
    plt.close()

    print(f"Modality visualization plots saved to ./{OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------------------------
def run_ablation_study(train_loader_raw, val_loader_raw, test_loader_raw, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading frozen encoders for embedding extraction...")
    encoder_model = MultimodalSystem(freeze_encoders=True).to(device)

    print("Extracting train embeddings...")
    train_img, train_text, train_labs, train_labels = extract_embeddings(encoder_model, train_loader_raw, device)
    print("Extracting validation embeddings...")
    val_img, val_text, val_labs, val_labels = extract_embeddings(encoder_model, val_loader_raw, device)
    print("Extracting test embeddings...")
    test_img, test_text, test_labs, test_labels = extract_embeddings(encoder_model, test_loader_raw, device)

    train_ds = CachedEmbeddingDataset(train_img, train_text, train_labs, train_labels)
    val_ds = CachedEmbeddingDataset(val_img, val_text, val_labs, val_labels)
    test_ds = CachedEmbeddingDataset(test_img, test_text, test_labs, test_labels)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
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
    trained_heads = {}
    all_histories = {}

    for name, head in conditions.items():
        print(f"\n{'=' * 60}\nTraining condition: {name}\n{'=' * 60}")
        head, history = train_head(head, train_loader, val_loader, device)
        probs, preds, labels = evaluate_head(head, test_loader, device)
        trained_heads[name] = head
        all_histories[name] = history

        report = classification_report(labels, preds, target_names=["Normal", "Pneumonia"], digits=4)
        print(report)
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "and")
        with open(f"{OUTPUT_DIR}/{safe_name}_classification_report.txt", "w") as f:
            f.write(report)

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

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        results[name] = {
            "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "auc": roc_auc, "fpr": fpr, "tpr": tpr,
        }

    plot_training_curves(all_histories)

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

    visualize_modality_dynamics(trained_heads["Vision + Text + Labs (Proposed)"], test_loader, device)

    return results


if __name__ == "__main__":
    DATASET_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed"

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Processed dataset not found at {DATASET_PATH}.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    dataset = load_from_disk(DATASET_PATH)

    split_ds = dataset.train_test_split(test_size=0.2, seed=42)
    train_val_ds = split_ds["train"]
    test_ds = split_ds["test"]

    sub_split = train_val_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = sub_split["train"]
    val_ds = sub_split["test"]

    columns_to_load = ["pixel_values", "input_ids", "attention_mask", "labels", "wbc", "crp"]
    train_ds.set_format(type="torch", columns=columns_to_load)
    val_ds.set_format(type="torch", columns=columns_to_load)
    test_ds.set_format(type="torch", columns=columns_to_load)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Test samples: {len(test_ds)}")

    train_loader_raw = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader_raw = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader_raw = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    run_ablation_study(train_loader_raw, val_loader_raw, test_loader_raw, device)