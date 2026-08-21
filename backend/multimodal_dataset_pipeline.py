import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer

# Configuration
os.environ["HF_HOME"] = "/Users/mickman/Documents/programs/PulmoNetAI"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
SAVE_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed"
PHYSIONET_DIR = "/Users/mickman/Documents/programs/PulmoNetAI/data/physionet_sepsis_psv"

FEATURE_COLS = ["HR", "O2Sat", "Temp", "Resp", "SBP", "MAP", "WBC", "FiO2"]
SEQ_LEN = 24

# Set Hugging Face Access Token from environment
HF_TOKEN = os.getenv("HF_TOKEN")


def load_and_preprocess_physionet_pool(psv_dir):
    """Loads PhysioNet PSV files, applies forward/back fills, and standardises parameters globally."""
    psv_files = sorted(glob.glob(os.path.join(psv_dir, "*.psv")))
    
    # Raise hard error instead of silent zero fallback
    if not psv_files:
        raise FileNotFoundError(
            f"No .psv files found in '{psv_dir}'. Run AWS CLI or extract PSV files before processing."
        )

    patient_tensors = []
    for path in psv_files[:2000]:
        df = pd.read_csv(path, sep="|")[FEATURE_COLS].copy()
        df = df.ffill().bfill().fillna(0.0)

        if len(df) >= SEQ_LEN:
            tensor = df.iloc[:SEQ_LEN].values
        else:
            pad_len = SEQ_LEN - len(df)
            padding = pd.DataFrame(0.0, index=np.arange(pad_len), columns=FEATURE_COLS)
            tensor = pd.concat([padding, df], ignore_index=True).values

        patient_tensors.append(tensor)

    pool = np.array(patient_tensors, dtype=np.float32)

    # Standardise across (N * 24, features)
    n_samples, seq_l, n_feats = pool.shape
    scaler = StandardScaler()
    pool_reshaped = scaler.fit_transform(pool.reshape(-1, n_feats))
    return pool_reshaped.reshape(n_samples, seq_l, n_feats)


if os.path.exists(SAVE_PATH) and os.listdir(SAVE_PATH):
    print("Loading processed dataset from local disk...")
    ds_final = load_from_disk(SAVE_PATH)
else:
    print("Local dataset not found. Processing from Hugging Face Hub...")

    # 1. Pre-load physiological pool with strict validation
    vitals_pool = load_and_preprocess_physionet_pool(PHYSIONET_DIR)
    num_pool_samples = len(vitals_pool)

    # 2. Download base dataset with auth token
    repo_id = "electricsheepafrica/Multimodal-Chest-X-ray-dataset-for-Normal-and-Bacterial-Pneumonia-in-Africans"
    ds = load_dataset(repo_id, token=HF_TOKEN)

    # 3. Download and merge metadata CSV deterministically
    metadata_path = hf_hub_download(
        repo_id=repo_id,
        filename="Comprehensive_Metadata.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
    metadata_df = pd.read_csv(metadata_path)

    # Add global explicit index column to bypass sharding index shifts
    ds_train = ds["train"].add_column("global_idx", list(range(len(ds["train"]))))

    def merge_metadata(example):
        idx = example["global_idx"]
        row = metadata_df.iloc[idx]
        example["Notes"] = row["Notes"]
        example["WBC_Count"] = row["WBC_Count"]
        example["CRP_Level"] = row["CRP_Level"]
        example["vitals_seq"] = vitals_pool[idx % num_pool_samples].tolist()
        return example

    print("Merging metadata and physiological time-series...")
    ds_merged = ds_train.map(merge_metadata, batched=False)

    # 4. Initialize tokenizers and processors
    img_processor = AutoImageProcessor.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256", token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", token=HF_TOKEN
    )

    # 5. Multimodal transformation function
    def transform(examples):
        pixel_values = img_processor(
            [img.convert("RGB") for img in examples["image"]], return_tensors="pt"
        )["pixel_values"]

        texts = [
            f"Notes: {n}. WBC: {w}. CRP: {c}."
            for n, w, c in zip(
                examples["Notes"], examples["WBC_Count"], examples["CRP_Level"]
            )
        ]
        inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": torch.tensor(examples["label"]).float().unsqueeze(1),
            "wbc": torch.tensor(examples["WBC_Count"]).float(),
            "crp": torch.tensor(examples["CRP_Level"]).float(),
            "vitals": torch.tensor(examples["vitals_seq"]).float(),
            "text_raw": texts,
        }

    print("Applying multimodal transformations...")
    ds_final = ds_merged.map(
        transform, batched=True, remove_columns=ds_merged.column_names
    )

    # 6. Save processed dataset locally
    os.makedirs(SAVE_PATH, exist_ok=True)
    ds_final.save_to_disk(SAVE_PATH)
    print(f"Dataset successfully saved to {SAVE_PATH}")