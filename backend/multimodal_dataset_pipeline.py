import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
os.environ["HF_HOME"] = "/Users/mickman/Documents/programs/PulmoNetAI"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
SAVE_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed"

# Check if pre-processed dataset already exists locally
if os.path.exists(SAVE_PATH) and os.listdir(SAVE_PATH):
    print("Loading processed dataset from local disk...")
    ds_final = load_from_disk(SAVE_PATH)

else:
    print("Local dataset not found. Downloading and processing from Hugging Face Hub...")
    
    # 1. Download the base dataset
    repo_id = "electricsheepafrica/Multimodal-Chest-X-ray-dataset-for-Normal-and-Bacterial-Pneumonia-in-Africans"
    ds = load_dataset(repo_id)
    
    # 2. Download and read metadata CSV
    metadata_path = hf_hub_download(repo_id=repo_id, filename="Comprehensive_Metadata.csv", repo_type="dataset")
    metadata_df = pd.read_csv(metadata_path)

    # 3. Merge metadata function
    def merge_metadata(example, idx):
        row = metadata_df.iloc[idx]
        example['Notes'] = row['Notes']
        example['WBC_Count'] = row['WBC_Count']
        example['CRP_Level'] = row['CRP_Level']
        return example

    print("Merging metadata...")
    ds_merged = ds['train'].map(merge_metadata, with_indices=True)

    # 4. Initialize tokenizers and processors
    img_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # 5. Transform function updated to retain and pass raw numerical parameters
    def transform(examples):
        pixel_values = img_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")["pixel_values"]
        
        texts = [f"Notes: {n}. WBC: {w}. CRP: {c}." for n, w, c in zip(examples['Notes'], examples['WBC_Count'], examples['CRP_Level'])]
        inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": torch.tensor(examples["label"]).float().unsqueeze(1),
            "wbc": torch.tensor(examples["WBC_Count"]).float(),
            "crp": torch.tensor(examples["CRP_Level"]).float(),
            "text_raw": texts 
        }

    print("Applying multimodal transformations...")
    ds_final = ds_merged.map(transform, batched=True, remove_columns=ds_merged.column_names)

    # 6. Save the processed dataset locally
    os.makedirs(SAVE_PATH, exist_ok=True)
    ds_final.save_to_disk(SAVE_PATH)
    print(f"Dataset successfully saved to {SAVE_PATH}")