"""
Attention weight distribution for the trained production model
(models/multimodal_pneumonia_model.pth).

Unlike ablation_study.py's visualize_modality_dynamics() -- which trains its
own ablation heads from scratch before it can plot anything -- this loads the
already-trained checkpoint via models.inference.load_model() and runs a
single inference pass over the held-out test split. No training involved, so
it's far cheaper to run.

Reduction and plots follow the same approach as
ablation_study.py::visualize_modality_dynamics(): each sample's self-attention
matrix (query token x key token, averaged over heads) is averaged over the
query dimension to get a per-modality "how much was this modality attended
to" score, matching how summarize_attention() in agent/utils.py reduces the
same tensor for the production report pipeline.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from models.inference import DEVICE, load_model

OUTPUT_DIR = "attention_results"
DATASET_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed"
BATCH_SIZE = 16

# Must match the joint_sequence concatenation order in
# MultimodalSystem.forward(): [img_vector, text_vector, lab_token, vitals_token]
MODALITY_ORDER = ["Vision", "Text", "Labs", "Vitals"]


def load_test_set():
    dataset = load_from_disk(DATASET_PATH)
    test_ds = dataset.train_test_split(test_size=0.2, seed=42)["test"]
    columns = ["pixel_values", "input_ids", "attention_mask", "labels", "wbc", "crp", "vitals"]
    test_ds.set_format(type="torch", columns=columns)
    return test_ds


def collect_attention_weights(model, loader):
    all_weights, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(DEVICE)
            text_input = {
                "input_ids": batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
            }
            labs_tensor = torch.stack([batch["wbc"], batch["crp"]], dim=-1).float().to(DEVICE)
            vitals_tensor = batch["vitals"].float().to(DEVICE)

            _, attn_weights = model(
                images=images,
                text_input=text_input,
                labs_tensor=labs_tensor,
                vitals_tensor=vitals_tensor,
                return_attention=True,
            )
            # attn_weights: [B, 4, 4] -> average over query dim -> [B, 4] per-modality score
            modality_importance = attn_weights.mean(dim=1)

            all_weights.append(modality_importance.cpu().numpy())
            all_labels.extend(batch["labels"].numpy().flatten())

    return np.vstack(all_weights), np.array(all_labels)


def plot_attention_distribution(weights, labels):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    weights_by_modality = {name: weights[:, i] for i, name in enumerate(MODALITY_ORDER)}

    print("\n" + "=" * 60)
    print("MODALITY SELF-ATTENTION ANALYSIS -- Production Model (Test Set)")
    print("=" * 60)
    for name, w in weights_by_modality.items():
        print(f"Mean {name} Attention: {np.mean(w) * 100:.2f}% (std: {np.std(w):.4f})")

    norm_mask = labels == 0
    pneu_mask = labels == 1
    print("\n  Normal cases:")
    for name, w in weights_by_modality.items():
        print(f"    {name}: {np.mean(w[norm_mask]) * 100:.2f}%")
    print("  Pneumonia cases:")
    for name, w in weights_by_modality.items():
        print(f"    {name}: {np.mean(w[pneu_mask]) * 100:.2f}%")

    row_sums = weights.sum(axis=1)
    print(f"\n  Sanity check -- mean row sum (should be ~1.0): {row_sums.mean():.4f}")

    plt.figure(figsize=(9, 5))
    colors = ["royalblue", "darkorange", "crimson", "seagreen"]
    for (name, w), color in zip(weights_by_modality.items(), colors):
        sns.kdeplot(w, label=f"{name} Token Attention Weight", fill=True, alpha=0.3, color=color)
    plt.title("Self-Attention Weight Distribution Across All Modalities (Production Model)")
    plt.xlabel("Attention Weight Score (0.0 to 1.0)")
    plt.ylabel("Density")
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attention_distribution.png", dpi=150)
    plt.close()

    df = pd.DataFrame(weights_by_modality)
    df["Diagnosis"] = ["Normal" if l == 0 else "Pneumonia" for l in labels]
    df_long = df.melt(id_vars="Diagnosis", var_name="Modality", value_name="Attention Weight")

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        x="Modality", y="Attention Weight", hue="Diagnosis",
        data=df_long, palette=["mediumseagreen", "indianred"],
    )
    plt.title("Modality Self-Attention Score by Diagnosis Class (Production Model)")
    plt.ylabel("Learned Self-Attention Weight")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attention_by_class.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to ./{OUTPUT_DIR}/")


def main():
    print(f"Using device: {DEVICE}")

    print("Loading trained model checkpoint...")
    model = load_model()

    print("Loading test set...")
    test_ds = load_test_set()
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    print("Running inference and collecting attention weights...")
    weights, labels = collect_attention_weights(model, loader)

    plot_attention_distribution(weights, labels)


if __name__ == "__main__":
    main()
