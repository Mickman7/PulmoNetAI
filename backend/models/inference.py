"""Loads the trained model + processors once (cached), and runs a single prediction.
Updated: forward() now requires labs_tensor (WBC/CRP via the 1D-CNN branch),
so predict() builds that tensor from the wbc/crp values passed in."""

import os
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

from .multimodal_system import MultimodalSystem, IMAGE_MODEL_NAME, TEXT_MODEL_NAME

MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "multimodal_pneumonia_model.pth")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


@lru_cache(maxsize=1)
def load_model():
    model = MultimodalSystem(freeze_encoders=True)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model


@lru_cache(maxsize=1)
def load_processors():
    img_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    return img_processor, tokenizer


def predict(image_path: str, notes: str, wbc: float, crp: float) -> dict:
    model = load_model()
    img_processor, tokenizer = load_processors()

    img = Image.open(image_path).convert("RGB")
    pixel_values = img_processor(img, return_tensors="pt")["pixel_values"].to(DEVICE)

    text_inputs = tokenizer(notes, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

    labs_tensor = torch.tensor([[wbc, crp]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits, attn_weights = model(pixel_values, text_inputs, labs_tensor, return_attention=True)
        probability = torch.sigmoid(logits).item()

    label = "Pneumonia" if probability > 0.5 else "Normal"
    return {"probability": probability, "label": label, "attn_weights": attn_weights}