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
VITALS_CHANNELS = 8  # HR, O2Sat, Temp, Resp, SBP, MAP, WBC, FiO2 -- must match training order

# Fixed label order for the joint self-attention sequence -- used everywhere
# a modality index needs to map back to a name (must match the order tokens
# are concatenated in VisionTextLabsVitalsHead.forward()).
MODALITY_ORDER = ["Vision", "Text", "Labs", "Vitals"]


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
    """Extracts image/text embeddings (via frozen encoders) plus the raw
    labs and vitals tensors (these bypass the frozen encoders entirely --
    they're processed by the trainable lab_cnn/vitals_cnn inside each head).

    NOTE: `batch["vitals"]` assumes your processed dataset stores a single
    [24, 8] array per sample under that column name. Verify this matches
    your actual dataset schema (from multimodal_dataset_pipeline.py) and
    adjust the key if it differs -- this could not be confirmed directly.
    """
    model.eval()
    img_seqs, text_pooled, labs, vitals, labels = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device)
            text_input = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            labs_tensor = torch.stack([batch["wbc"], batch["crp"]], dim=-1).float().to(device)
            vitals_tensor = batch["vitals"].float().to(device)  # expected shape [B, 24, VITALS_CHANNELS]

            img_features = model.image_encoder(images).last_hidden_state
            text_features = model.text_encoder(**text_input).pooler_output

            img_seqs.append(img_features.cpu())
            text_pooled.append(text_features.cpu())
            labs.append(labs_tensor.cpu())
            vitals.append(vitals_tensor.cpu())
            labels.append(batch["labels"].cpu())

    return (
        torch.cat(img_seqs),
        torch.cat(text_pooled),
        torch.cat(labs),
        torch.cat(vitals),
        torch.cat(labels),
    )


class CachedEmbeddingDataset(Dataset):
    def __init__(self, img_seq, text_pooled, labs, vitals, labels):
        self.img_seq = img_seq
        self.text_pooled = text_pooled
        self.labs = labs
        self.vitals = vitals
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.img_seq[idx], self.text_pooled[idx], self.labs[idx],
            self.vitals[idx], self.labels[idx],
        )


# ---------------------------------------------------------------------------
# 3. Model Heads
# ---------------------------------------------------------------------------
class VisionOnlyHead(nn.Module):
    def __init__(self, img_hidden_size, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.img_proj = nn.Linear(img_hidden_size, embed_dim)
        self.img_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq, text_pooled=None, labs=None, vitals=None, **kwargs):
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

    def forward(self, img_seq=None, text_pooled=None, labs=None, vitals=None, **kwargs):
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

        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.project = nn.Linear(embed_dim * 2, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq, text_pooled, labs=None, vitals=None, **kwargs):
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


class VisionTextLabsVitalsHead(nn.Module):
    """The 'Proposed' architecture -- now matching your production
    MultimodalSystem exactly: all four modalities (image, text, labs,
    vitals) are pooled to single tokens, concatenated into one 4-token
    sequence, and fused via SELF-attention (Q=K=V=joint_sequence), NOT
    cross-attention. This replaces the previous 2-token cross-attention
    design (text as Query, [image, labs] as Key/Value)."""

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

        self.vitals_cnn = nn.Sequential(
            nn.Conv1d(in_channels=VITALS_CHANNELS, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.vitals_proj = nn.Linear(32, embed_dim)

        self.img_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.lab_norm = nn.LayerNorm(embed_dim)
        self.vitals_norm = nn.LayerNorm(embed_dim)

        # Self-attention over the joint 4-token sequence, matching production
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img_seq, text_pooled, labs, vitals=None, return_visuals=False):
        batch_size = img_seq.size(0)

        img_vector = self.img_norm(self.img_proj(img_seq)).mean(dim=1, keepdim=True)     # [B, 1, embed_dim]
        text_vector = self.text_norm(self.text_proj(text_pooled)).unsqueeze(1)           # [B, 1, embed_dim]

        labs_formatted = labs.unsqueeze(1) if labs.dim() == 2 else labs
        lab_features = self.lab_cnn(labs_formatted).squeeze(-1)
        lab_token = self.lab_norm(self.lab_proj(lab_features)).unsqueeze(1)              # [B, 1, embed_dim]

        if vitals is not None:
            vitals_formatted = vitals.transpose(1, 2)  # [B, 24, C] -> [B, C, 24]
            vitals_features = self.vitals_cnn(vitals_formatted).squeeze(-1)
            vitals_vector = self.vitals_norm(self.vitals_proj(vitals_features))
        else:
            vitals_vector = torch.zeros((batch_size, img_vector.size(-1)), device=img_seq.device, dtype=img_vector.dtype)
        vitals_token = vitals_vector.unsqueeze(1)  # [B, 1, embed_dim]

        # Order here MUST match MODALITY_ORDER at the top of this file
        joint_sequence = torch.cat([img_vector, text_vector, lab_token, vitals_token], dim=1)  # [B, 4, embed_dim]

        attn_out, attn_weights = self.fusion(
            joint_sequence, joint_sequence, joint_sequence,
            need_weights=True, average_attn_weights=True,
        )
        # attn_weights shape: [B, 4, 4] -- row=query token, col=key token, averaged over heads

        pooled_sequence = attn_out.mean(dim=1)  # [B, embed_dim]
        fused = self.norm(pooled_sequence)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        if return_visuals:
            return logits, attn_weights

        return logits


# ---------------------------------------------------------------------------
# 4. Training and Evaluation Loops
# ---------------------------------------------------------------------------
def _compute_val_loss(head, val_loader, device, criterion):
    head.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img_seq, text_pooled, labs, vitals, labels in val_loader:
            img_seq, text_pooled, labs, vitals = (
                img_seq.to(device), text_pooled.to(device), labs.to(device), vitals.to(device),
            )
            labels = labels.to(device).float()
            logits = head(img_seq, text_pooled, labs, vitals)
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
        for img_seq, text_pooled, labs, vitals, labels in train_loader:
            img_seq, text_pooled, labs, vitals = (
                img_seq.to(device), text_pooled.to(device), labs.to(device), vitals.to(device),
            )
            labels = labels.to(device).float()

            optimiser.zero_grad()
            logits = head(img_seq, text_pooled, labs, vitals)
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
        for img_seq, text_pooled, labs, vitals, labels in test_loader:
            img_seq, text_pooled, labs, vitals = (
                img_seq.to(device), text_pooled.to(device), labs.to(device), vitals.to(device),
            )
            logits = head(img_seq, text_pooled, labs, vitals)
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
# 5. Extract and Plot 4-Modality Self-Attention Weights
# ---------------------------------------------------------------------------
def visualize_modality_dynamics(proposed_head, test_loader, device):
    """Extracts and visualises self-attention weights across ALL FOUR
    modality tokens (Vision, Text, Labs, Vitals).

    attn_weights from the model is [B, 4, 4] (query token x key token,
    averaged over heads). To get a single "how much was modality X relied
    on" score per sample, we average over the QUERY dimension (dim=1) --
    i.e. across all 4 tokens' queries, how much attention did each key
    token (column) receive on average. This mirrors summarize_attention()
    in backend/models/inference.py, which does the same reduction for the
    production model.
    """
    proposed_head.eval()
    proposed_head.to(device)

    all_attn_weights = []  # will hold per-sample [4] importance vectors
    all_labels = []

    with torch.no_grad():
        for img_seq, text_pooled, labs, vitals, labels in test_loader:
            img_seq = img_seq.to(device)
            text_pooled = text_pooled.to(device)
            labs = labs.to(device)
            vitals = vitals.to(device)

            _, attn_weights = proposed_head(img_seq, text_pooled, labs, vitals, return_visuals=True)
            # attn_weights: [B, 4, 4] -> average over query dim -> [B, 4]
            modality_importance = attn_weights.mean(dim=1)

            all_attn_weights.append(modality_importance.cpu().numpy())
            all_labels.extend(labels.numpy().flatten())

    all_attn_weights = np.vstack(all_attn_weights)  # [N_samples, 4]
    all_labels = np.array(all_labels)

    weights_by_modality = {name: all_attn_weights[:, i] for i, name in enumerate(MODALITY_ORDER)}

    print("\n" + "=" * 60)
    print("MODALITY SELF-ATTENTION ANALYSIS (Test Set, 4 modalities)")
    print("=" * 60)
    for name, weights in weights_by_modality.items():
        print(f"Mean {name} Attention: {np.mean(weights) * 100:.2f}% (std: {np.std(weights):.4f})")

    norm_mask = all_labels == 0
    pneu_mask = all_labels == 1

    print("\n  Normal cases:")
    for name, weights in weights_by_modality.items():
        print(f"    {name}: {np.mean(weights[norm_mask]) * 100:.2f}%")
    print("  Pneumonia cases:")
    for name, weights in weights_by_modality.items():
        print(f"    {name}: {np.mean(weights[pneu_mask]) * 100:.2f}%")

    # Sanity check -- each sample's 4 weights should sum to ~1.0 since they're
    # an average over rows of an attention matrix (each row sums to 1.0)
    row_sums = all_attn_weights.sum(axis=1)
    print(f"\n  Sanity check -- mean row sum (should be ~1.0): {row_sums.mean():.4f}")

    # Plot 1: distribution of attention weight per modality
    plt.figure(figsize=(9, 5))
    colors = ["royalblue", "darkorange", "crimson", "seagreen"]
    for (name, weights), color in zip(weights_by_modality.items(), colors):
        sns.kdeplot(weights, label=f"{name} Token Attention Weight", fill=True, alpha=0.3, color=color)
    plt.title("Self-Attention Weight Distribution Across All Modalities")
    plt.xlabel("Attention Weight Score (0.0 to 1.0)")
    plt.ylabel("Density")
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sequence_attention_distribution.png", dpi=150)
    plt.close()

    # Plot 2: per-modality attention weight by diagnosis class (one panel per modality)
    df_attn = pd.DataFrame(weights_by_modality)
    df_attn["Diagnosis"] = ["Normal" if l == 0 else "Pneumonia" for l in all_labels]
    df_long = df_attn.melt(id_vars="Diagnosis", var_name="Modality", value_name="Attention Weight")

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        x="Modality", y="Attention Weight", hue="Diagnosis",
        data=df_long, palette=["mediumseagreen", "indianred"],
    )
    plt.title("Modality Self-Attention Score by Diagnosis Class")
    plt.ylabel("Learned Self-Attention Weight")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attention_by_class.png", dpi=150)
    plt.close()

    print(f"Modality visualisation plots saved to ./{OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------------------------
def run_ablation_study(train_loader_raw, val_loader_raw, test_loader_raw, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading frozen encoders for embedding extraction...")
    encoder_model = MultimodalSystem(freeze_encoders=True).to(device)

    print("Extracting train embeddings...")
    train_img, train_text, train_labs, train_vitals, train_labels = extract_embeddings(encoder_model, train_loader_raw, device)
    print("Extracting validation embeddings...")
    val_img, val_text, val_labs, val_vitals, val_labels = extract_embeddings(encoder_model, val_loader_raw, device)
    print("Extracting test embeddings...")
    test_img, test_text, test_labs, test_vitals, test_labels = extract_embeddings(encoder_model, test_loader_raw, device)

    train_ds = CachedEmbeddingDataset(train_img, train_text, train_labs, train_vitals, train_labels)
    val_ds = CachedEmbeddingDataset(val_img, val_text, val_labs, val_vitals, val_labels)
    test_ds = CachedEmbeddingDataset(test_img, test_text, test_labs, test_vitals, test_labels)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    img_hidden = encoder_model.image_encoder.config.hidden_size
    text_hidden = encoder_model.text_encoder.config.hidden_size

    conditions = {
        "Vision Only": VisionOnlyHead(img_hidden),
        "Text Only": TextOnlyHead(text_hidden),
        "Vision + Text": VisionTextHead(img_hidden, text_hidden),
        "Vision + Text + Labs + Vitals (Proposed)": VisionTextLabsVitalsHead(img_hidden, text_hidden),
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

    visualize_modality_dynamics(trained_heads["Vision + Text + Labs + Vitals (Proposed)"], test_loader, device)

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

    # NOTE: "vitals" column name assumed -- verify against your actual
    # processed dataset schema and adjust if it differs.
    columns_to_load = ["pixel_values", "input_ids", "attention_mask", "labels", "wbc", "crp", "vitals"]
    train_ds.set_format(type="torch", columns=columns_to_load)
    val_ds.set_format(type="torch", columns=columns_to_load)
    test_ds.set_format(type="torch", columns=columns_to_load)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Test samples: {len(test_ds)}")

    train_loader_raw = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader_raw = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader_raw = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    run_ablation_study(train_loader_raw, val_loader_raw, test_loader_raw, device)