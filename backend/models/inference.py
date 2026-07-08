"""
Single entrypoint for running the model. Both the /predict route and the
agent (which needs probability + attention weights) call into predict()
so preprocessing logic isn't duplicated across the codebase.
"""

from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

from .multimodal_system import MultimodalSystem, IMAGE_MODEL_NAME, TEXT_MODEL_NAME

MODEL_WEIGHTS_PATH = "backend/models/multimodal_pneumonia_model.pth"  # <-- update to your .pth


@lru_cache(maxsize=1)
def load_model() -> MultimodalSystem:
    model = MultimodalSystem(freeze_encoders=True)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model


@lru_cache(maxsize=1)
def load_processors():
    img_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    return img_processor, tokenizer


def predict(image_path: str, notes: str, wbc: float, crp: float) -> dict:
    """Runs the full forward pass and returns everything downstream consumers need."""
    model = load_model()
    img_processor, tokenizer = load_processors()

    img = Image.open(image_path).convert("RGB")
    pixel_values = img_processor(img, return_tensors="pt")["pixel_values"]

    text_str = f"Notes: {notes}. WBC: {wbc}. CRP: {crp}."
    text_inputs = tokenizer(text_str, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        logits, attn_weights = model(pixel_values, text_inputs, return_attention=True)
        probability = torch.sigmoid(logits).item()

    return {
        "probability": probability,
        "label": "Pneumonia" if probability > 0.5 else "Normal",
        "attn_weights": attn_weights,  # kept in-memory only, not persisted to DB
    }
