# test_eval.py
import argparse
import os
from datasets import load_from_disk
from eval_metrics import run_evaluation
from model import MultimodalSystem
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(
        description="PulmoNetAI Standalone Custom Modality Evaluation Script"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/Users/mickman/Documents/programs/PulmoNetAI/data/ds_full_processed",
        help="Path to the processed dataset directory",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/Users/mickman/Documents/programs/PulmoNetAI/backend/models/multimodal_pneumonia_model.pth",
        help="Path to the saved model state dict (.pth)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Evaluation batch size"
    )
    parser.add_argument(
        "--test_split_size",
        type=float,
        default=0.2,
        help="Test split ratio used in train_test_split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train_test_split reproducibility",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vision_only", "text_only", "bimodal", "trimodal"],
        choices=["vision_only", "text_only", "bimodal", "trimodal"],
        help="List of modality modes to evaluate",
    )

    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Processed dataset not found at {args.dataset_path}."
        )

    print(f"Loading processed dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    split_ds = dataset.train_test_split(
        test_size=args.test_split_size, seed=args.seed
    )
    test_ds = split_ds["test"]

    columns_to_load = [
        "pixel_values",
        "input_ids",
        "attention_mask",
        "labels",
        "wbc",
        "crp",
    ]
    test_ds.set_format(type="torch", columns=columns_to_load)

    print(f"Test samples loaded: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = MultimodalSystem(freeze_encoders=True).to(device)

    for mode in args.modes:
        print(f"\n==================================================")
        print(f"RUNNING EVALUATION MODE: {mode.upper()}")
        print(f"==================================================")

        run_evaluation(
            model=model,
            test_loader=test_loader,
            device=device,
            model_path=args.model_path,
            modality_mode=mode,
        )


if __name__ == "__main__":
    main()