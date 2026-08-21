import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_from_disk

SAVE_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed"
FEATURE_COLS = ["HR", "O2Sat", "Temp", "Resp", "SBP", "MAP", "WBC", "FiO2"]

def inspect_dataset(save_path):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"No processed dataset found at {save_path}")

    ds = load_from_disk(save_path)
    print(f"Dataset Loaded Successfully!")
    print(f"Total Samples: {len(ds)}")
    print(f"Schema Features: {list(ds.features.keys())}\n")

    # 1. Format inspection
    sample = ds[0]
    print("--- Sample 0 Field Breakdown ---")
    for key, val in sample.items():
        if isinstance(val, list):
            tensor_val = torch.tensor(val)
            print(f"{key:15s} | Type: List -> Tensor | Shape: {list(tensor_val.shape)} | Dtype: {tensor_val.dtype}")
        elif isinstance(val, str):
            print(f"{key:15s} | Type: String         | Content Preview: {val[:60]}...")
        else:
            print(f"{key:15s} | Type: {type(val).__name__:12s} | Value: {val}")

    print("\n--- Tensor Shape Consistency Check ---")
    pixel_shape = torch.tensor(sample["pixel_values"]).shape
    input_ids_shape = torch.tensor(sample["input_ids"]).shape
    vitals_shape = torch.tensor(sample["vitals"]).shape
    
    print(f"pixel_values : {list(pixel_shape)} (Expected: [3, 256, 256])")
    print(f"input_ids    : {list(input_ids_shape)} (Expected: [128])")
    print(f"vitals       : {list(vitals_shape)} (Expected: [24, 8])")

    # 2. Visual Inspection
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Display image (Denormalise PyTorch tensor to RGB)
    img_tensor = torch.tensor(sample["pixel_values"]).numpy()
    img_disp = np.transpose(img_tensor, (1, 2, 0))
    img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())
    
    axes[0].imshow(img_disp)
    axes[0].set_title(f"Label: {sample['labels']} | WBC: {sample['wbc']} | CRP: {sample['crp']}")
    axes[0].axis("off")

    # Display Vitals Sequence Heatmap
    vitals_matrix = np.array(sample["vitals"]).T  # Shape: (8 features, 24 hours)
    sns.heatmap(
        vitals_matrix,
        ax=axes[1],
        cmap="viridis",
        yticklabels=FEATURE_COLS,
        cbar_kws={'label': 'Standardised Value'}
    )
    axes[1].set_title("24-Hour Vitals Sequence (Normalized)")
    axes[1].set_xlabel("Hour Index (0-23)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    inspect_dataset(SAVE_PATH)