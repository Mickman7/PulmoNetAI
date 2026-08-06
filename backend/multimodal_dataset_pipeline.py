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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

# Configuration
os.environ["HF_HOME"] = "/Users/mickman/Documents/programs/PulmoNetAI"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
SAVE_PATH = (
    "/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed"
)
PHYSIONET_DIR = "/Users/mickman/Documents/programs/PulmoNetAI/data/physionet_sepsis_psv"  # Path to your .psv files

FEATURE_COLS = ["HR", "O2Sat", "Temp", "Resp", "SBP", "MAP", "WBC", "FiO2"]
SEQ_LEN = 24


def load_and_preprocess_physionet_pool(psv_dir):
  """Loads all PhysioNet PSV files, forward/back fills NaNs, and normalises them."""
  psv_files = sorted(glob.glob(os.path.join(psv_dir, "*.psv")))
  if not psv_files:
    print(
      "Warning: No PSV files found in directory. Generating zero placeholder pool."
    )
    return np.zeros((100, SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32)

  patient_tensors = []
  for path in psv_files[:2000]:  # Pool subset for fast indexing
    df = pd.read_csv(path, sep="|")[FEATURE_COLS].copy()
    df = df.ffill().bfill().fillna(0.0)

    if len(df) >= SEQ_LEN:
      tensor = df.iloc[:SEQ_LEN].values
    else:
      pad_len = SEQ_LEN - len(df)
      padding = pd.DataFrame(
          0.0, index=np.arange(pad_len), columns=FEATURE_COLS
      )
      tensor = pd.concat([padding, df], ignore_index=True).values

    patient_tensors.append(tensor)

  pool = np.array(patient_tensors, dtype=np.float32)  # Shape: (N, 24, 8)

  # Standardise globally across (N * 24, features)
  n_samples, seq_l, n_feats = pool.shape
  scaler = StandardScaler()
  pool_reshaped = scaler.fit_transform(pool.reshape(-1, n_feats))
  return pool_reshaped.reshape(n_samples, seq_l, n_feats)


# Check if pre-processed dataset already exists locally
if os.path.exists(SAVE_PATH) and os.listdir(SAVE_PATH):
  print("Loading processed dataset from local disk...")
  ds_final = load_from_disk(SAVE_PATH)
else:
  print(
      "Local dataset not found. Downloading and processing from Hugging Face"
      " Hub..."
  )

  # 1. Download base dataset
  repo_id = (
      "electricsheepafrica/Multimodal-Chest-X-ray-dataset-for-Normal-and-Bacterial-Pneumonia-in-Africans"
  )
  ds = load_dataset(repo_id)

  # 2. Pre-load physiological pool
  vitals_pool = load_and_preprocess_physionet_pool(PHYSIONET_DIR)
  num_pool_samples = len(vitals_pool)

  # 3. Download and read metadata CSV
  metadata_path = hf_hub_download(
      repo_id=repo_id,
      filename="Comprehensive_Metadata.csv",
      repo_type="dataset",
  )
  metadata_df = pd.read_csv(metadata_path)

  # Merge metadata function
  def merge_metadata(example, idx):
    row = metadata_df.iloc[idx]
    example["Notes"] = row["Notes"]
    example["WBC_Count"] = row["WBC_Count"]
    example["CRP_Level"] = row["CRP_Level"]
    # Assign deterministic index from vitals pool
    example["vitals_seq"] = vitals_pool[idx % num_pool_samples].tolist()
    return example

  print("Merging metadata and physiological time-series...")
  ds_merged = ds["train"].map(merge_metadata, with_indices=True)

  # 4. Initialize tokenizers and processors
  img_processor = AutoImageProcessor.from_pretrained(
      "microsoft/swinv2-tiny-patch4-window8-256"
  )
  tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

  # 5. Transform function updated with vitals sequence tensor
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
        "vitals": torch.tensor(examples["vitals_seq"]).float(),  # Shape: (Batch, 24, 8)
        "text_raw": texts,
    }

  print("Applying multimodal transformations...")
  ds_final = ds_merged.map(
      transform, batched=True, remove_columns=ds_merged.column_names
  )

  # 6. Save the processed dataset locally
  os.makedirs(SAVE_PATH, exist_ok=True)
  ds_final.save_to_disk(SAVE_PATH)
  print(f"Dataset successfully saved to {SAVE_PATH}")