import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def run_evaluation(model, test_loader, device):
    """
    Loads saved weights, evaluates the multimodal system, and plots results.
    """
    MODEL_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/multimodal_pneumonia_model.pth"
    
    # 1. Load the trained weights safely onto your current device (MPS/CPU)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model weights not found at {MODEL_PATH}. Save the model first.")
        
    print(f"Loading trained weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference on test dataset...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            text_input = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }

            # Inference pass using the complete model architecture
            logits = model(images, text_input)
            probs = torch.sigmoid(logits)      
            preds = (probs > 0.5).int()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # 2. Generate and display scannable classification metrics
    report = classification_report(
        all_labels,
        all_preds,
        target_names=['Normal', 'Pneumonia'],
        digits=4
    )

    print("\n" + "="*50)
    print("--- Multimodal System Classification Report ---")
    print("="*50)
    print(report)
    print("="*50 + "\n")

    # 3. Generate, plot, and save the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white", font_scale=1.2)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=['Normal', 'Pneumonia'],
        yticklabels=['Normal', 'Pneumonia']
    )
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix: Multimodal Pneumonia Detection', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save the output image locally for your documentation/frontend use
    os.makedirs("evaluation_results", exist_ok=True)
    plt.savefig("evaluation_results/confusion_matrix.png", dpi=300)
    print("Confusion matrix graphic saved to: evaluation_results/confusion_matrix.png")
    plt.show()

# To integrate this inside your main pipeline execution file, simply add:
# from evaluation import run_evaluation
# run_evaluation(model, test_loader, device)