# backend/evaluation.py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    roc_curve,
)


def run_evaluation(
    model, test_loader, device, model_path=None, modality_mode="trimodal"
):
    """Loads saved weights and evaluates specific sub-modalities by selectively

    passing or zero-masking input features.

    modality_mode options: 'vision_only', 'text_only', 'bimodal', 'trimodal'
    """
    if model_path is None:
        model_path = "/Users/mickman/Documents/programs/PulmoNetAI/backend/models/multimodal_pneumonia_model.pth"

    OUTPUT_DIR = "evaluation_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model weights not found at {model_path}.")

    print(f"\nLoading trained weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    error_cases = []

    print(f"Running inference for mode: [{modality_mode.upper()}]...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            labels = batch["labels"].to(device)

            # --- Selective Modality Masking ---
            # 1. Vision
            if modality_mode in ["vision_only", "bimodal", "trimodal"]:
                images = batch["pixel_values"].to(device)
            else:
                images = torch.zeros_like(batch["pixel_values"]).to(device)

            # 2. Text
            if modality_mode in ["text_only", "bimodal", "trimodal"]:
                text_input = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
            else:
                text_input = {
                    "input_ids": torch.zeros_like(batch["input_ids"]).to(
                        device
                    ),
                    "attention_mask": torch.zeros_like(
                        batch["attention_mask"]
                    ).to(device),
                }

            # 3. Labs (Static WBC/CRP)
            if modality_mode == "trimodal":
                labs_tensor = (
                    torch.stack([batch["wbc"], batch["crp"]], dim=-1)
                    .float()
                    .to(device)
                )
            else:
                labs_tensor = torch.zeros(
                    (batch["labels"].shape[0], 2), device=device
                )

            # 4. Vitals (24x8 Temporal Sequence)
            if modality_mode == "trimodal" and "vitals" in batch:
                vitals_tensor = batch["vitals"].float().to(device)
            else:
                # Shape matches (Batch, Seq_Len=24, Features=8)
                vitals_tensor = torch.zeros(
                    (batch["labels"].shape[0], 24, 8), device=device
                )

            # Model Forward Pass
            logits, attn_weights = model(
                images,
                text_input,
                labs_tensor=labs_tensor,
                vitals_tensor=vitals_tensor,
                return_attention=True,
            )
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_np)

            for i in range(len(labels_np)):
                if preds[i] != labels_np[i]:
                    error_cases.append({
                        "batch_index": batch_idx,
                        "sample_within_batch": i,
                        "true_label": int(labels_np[i]),
                        "pred_label": int(preds[i]),
                        "model_probability": float(probs[i]),
                        "attention_weights": (
                            attn_weights[i].cpu().numpy()
                            if attn_weights is not None
                            else None
                        ),
                        "raw_text_input": (
                            batch.get("text_raw", [""])[i]
                            if "text_raw" in batch
                            else "Text raw unmapped"
                        ),
                    })

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metric Extraction
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision_macro = precision_score(all_labels, all_preds, average="macro")

    print("\n" + "=" * 60)
    print(
        f"--- PERFORMANCE REPORT: {modality_mode.upper()} EVALUATION ---"
    )
    print("=" * 60)
    print(f"Overall Accuracy (%): {accuracy:.2f}%")
    print(f"Macro Precision:     {precision_macro:.4f}")
    print("-" * 60)
    print(
        f"{'Class':<12} | {'Precision':<10} | {'Recall (Sens)':<14} |"
        f" {'F1-Score':<10} | {'Volume':<8}"
    )
    print("-" * 60)
    print(
        f"{'Normal':<12} | {precision[0]:.4f}     | {recall[0]:.4f}        "
        f" | {f1[0]:.4f}   | {support[0]}"
    )
    print(
        f"{'Pneumonia':<12} | {precision[1]:.4f}     | {recall[1]:.4f}        "
        f" | {f1[1]:.4f}   | {support[1]}"
    )
    print("=" * 60)