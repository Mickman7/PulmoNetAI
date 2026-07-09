import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_fscore_support
)
from sklearn.calibration import calibration_curve

def run_evaluation(model, test_loader, device):
    """
    Loads saved weights, evaluates the multimodal system, and plots advanced 
    clinical diagnostics (ROC, Calibration, Per-Class breakdowns, and Error Case Analytics).
    """
    MODEL_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/backend/models/multimodal_pneumonia_model.pth"
    OUTPUT_DIR = "evaluation_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load the trained weights safely
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model weights not found at {MODEL_PATH}. Save the model first.")
        
    print(f"Loading trained weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    
    # Structures to log raw information for clinical error diagnostics
    error_cases = []

    print("Running inference on test dataset...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            text_input = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            
            # Extract and stack physiological lab variables for the 1D-CNN branch
            labs_tensor = torch.stack([batch["wbc"], batch["crp"]], dim=-1).float().to(device)

            # Extract predictions and multihead cross-attention mappings with all three required inputs
            logits, attn_weights = model(images, text_input, labs_tensor=labs_tensor, return_attention=True)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()    
            preds = (probs > 0.5).astype(int)
            labels_np = labels.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_np)

            # Isolate misclassifications inline for localized inspection
            for i in range(len(labels_np)):
                if preds[i] != labels_np[i]:
                    error_cases.append({
                        "batch_index": batch_idx,
                        "sample_within_batch": i,
                        "true_label": int(labels_np[i]),
                        "pred_label": int(preds[i]),
                        "model_probability": float(probs[i]),
                        "attention_weights": attn_weights[i].cpu().numpy() if attn_weights is not None else None,
                        "raw_text_input": batch.get("text_raw", [""])[i] if "text_raw" in batch else "Text raw unmapped"
                    })

    # Convert aggregated outputs to uniform arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # -----------------------------------------------------------------------
    # 2. Per-Class Metrics & Standard Classification Breakdown
    # -----------------------------------------------------------------------
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    print("\n" + "="*60)
    print("--- MULTIMODAL PNEUMONIA SYSTEM PERFORMANCE REPORT ---")
    print("="*60)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall (Sens)':<14} | {'F1-Score':<10} | {'Volume':<8}")
    print("-"*60)
    print(f"{'Normal':<12} | {precision[0]:.4f}     | {recall[0]:.4f}         | {f1[0]:.4f}   | {support[0]}")
    print(f"{'Pneumonia':<12} | {precision[1]:.4f}     | {recall[1]:.4f}         | {f1[1]:.4f}   | {support[1]}")
    print("="*60)

    # -----------------------------------------------------------------------
    # 3. Clinical Plotting Layer: Confusion Matrix, ROC, Calibration
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Subplot A: Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'], ax=axes[0])
    axes[0].set_xlabel('Predicted Label', fontweight='bold')
    axes[0].set_ylabel('Actual Label', fontweight='bold')
    axes[0].set_title('Confusion Matrix Diagnostics')

    # Subplot B: ROC Curve Plotting (AUC Evaluation)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    axes[1].set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    axes[1].set_title('Receiver Operating Characteristic (ROC)')
    axes[1].legend(loc="lower right")

    # Subplot C: Calibration Curve Execution (Reliability Tracking)
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10, strategy='uniform')
    axes[2].plot(prob_pred, prob_true, marker='s', lw=2, label='Multimodal System')
    axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    axes[2].set_xlabel('Mean Predicted Probability Confidence', fontweight='bold')
    axes[2].set_ylabel('Fraction of Real-World Positives', fontweight='bold')
    axes[2].set_title('Reliability Calibration Curve')
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/multimodal_diagnostics_panel.png", dpi=300)
    print(f"Core clinical visual diagnostic panel saved to: {OUTPUT_DIR}/multimodal_diagnostics_panel.png")
    plt.show()

    # -----------------------------------------------------------------------
    # 4. Comprehensive Clinical Error Case Inspection
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"--- CLINICAL ERROR ANALYSIS INSIGHTS (Total Misclassified: {len(error_cases)}) ---")
    print("="*60)
    
    if len(error_cases) == 0:
        print("Exceptional Performance: Zero verification error boundaries tripped across the test split.")
    else:
        borderline_errors = [e for e in error_cases if 0.4 <= e["model_probability"] <= 0.6]
        critical_errors = [e for e in error_cases if e["model_probability"] < 0.15 or e["model_probability"] > 0.85]
        
        print(f"-> Borderline Ambiguity Cases (0.4 <= P <= 0.6): {len(borderline_errors)}")
        print(f"-> Critical/Overconfident Divergence Cases (P < 0.15 or P > 0.85): {len(critical_errors)}")
        print("-"*60)
        
        print("\nDisplaying Sample Critical Error Trajectories:")
        for idx, case in enumerate(critical_errors[:3]):
            print(f"\n[Critical Failure Case #{idx + 1}]")
            print(f"  - System True Target Condition: {'Pneumonia' if case['true_label'] == 1 else 'Normal'}")
            print(f"  - Model Erroneous Prediction: {'Pneumonia' if case['pred_label'] == 1 else 'Normal'}")
            print(f"  - Output Sigmoid Probability: {case['model_probability']:.4%}")
            
            if case["attention_weights"] is not None:
                max_attn_val = case["attention_weights"].max()
                mean_attn_val = case["attention_weights"].mean()
                print(f"  - Cross-Attention Variance Peak: {max_attn_val:.4f} (Mean Baseline: {mean_attn_val:.4f})")
                if max_attn_val > 2.0 * mean_attn_val:
                    print("    * Diagnostics: Attention map is intensely focused on localized regions despite wrong prediction.")
                else:
                    print("    * Diagnostics: Attention map is highly diffuse, indicating high feature confusion across patches.")
            
            print(f"  - Input Sequence Data Captured: {case['raw_text_input'][:120]}...")
    print("="*60 + "\n")