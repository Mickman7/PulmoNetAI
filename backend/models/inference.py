from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

from .multimodal_system import MultimodalSystem, IMAGE_MODEL_NAME, TEXT_MODEL_NAME

MODEL_WEIGHTS_PATH = "backend/models/multimodal_pneumonia_model.pth"


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

    # 1. Image Preprocessing
    img = Image.open(image_path).convert("RGB")
    pixel_values = img_processor(img, return_tensors="pt")["pixel_values"]

    # 2. Text Preprocessing
    text_str = f"Notes: {notes}. WBC: {wbc}. CRP: {crp}."
    text_inputs = tokenizer(text_str, return_tensors="pt", padding=True, truncation=True)

    # 3. Lab Results Preprocessing for 1D-CNN Branch
    # Formats to a 2D floating-point tensor vector matching training configuration
    labs_tensor = torch.tensor([[float(wbc), float(crp)]]).float()

    with torch.no_grad():
        # Pass the explicit labs tensor into the updated forward call signature
        logits, attn_weights = model(pixel_values, text_inputs, labs_tensor=labs_tensor, return_attention=True)
        probability = torch.sigmoid(logits).item()

    return {
        "probability": probability,
        "label": "Pneumonia" if probability > 0.5 else "Normal",
        "attn_weights": attn_weights,
    }