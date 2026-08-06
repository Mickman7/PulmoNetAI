"""Loads the trained model + processors once (cached), and runs a single prediction.

Vitals handling: model.forward() now handles vitals_tensor=None internally
(falls back to a zero token) -- so we don't need to fabricate a fake tensor
here anymore. Pass real `vitals` (shape [24, VITALS_CHANNELS]) in whenever
you have them; otherwise the model treats vitals as simply unavailable.
"""

import os
from functools import lru_cache

import numpy as np
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


def predict(image_path: str, notes: str, wbc: float, crp: float, vitals=None) -> dict:
    """vitals, if provided, should be array-like of shape [24, VITALS_CHANNELS]."""
    model = load_model()
    img_processor, tokenizer = load_processors()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    img = Image.open(image_path).convert("RGB")
    pixel_values = img_processor(img, return_tensors="pt")["pixel_values"].to(DEVICE)

    text_inputs = tokenizer(notes, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

    safe_wbc = float(wbc) if wbc is not None else 0.0
    safe_crp = float(crp) if crp is not None else 0.0
    labs_tensor = torch.tensor([[safe_wbc, safe_crp]], dtype=torch.float32).to(DEVICE)

    vitals_tensor = None
    if vitals is not None:
        vitals_tensor = torch.tensor(np.asarray(vitals, dtype=np.float32)).unsqueeze(0).to(DEVICE)  # [1, 24, C]

    with torch.no_grad():
        logits, attn_weights = model(
            images=pixel_values,
            text_input=text_inputs,
            labs_tensor=labs_tensor,
            vitals_tensor=vitals_tensor,  # None is safe now -- handled inside forward()
            return_attention=True,
        )
        probability = torch.sigmoid(logits).item()

    label = "Pneumonia" if probability > 0.5 else "Normal"
    return {
        "probability": probability,
        "label": label,
        "attn_weights": attn_weights,
        "used_real_vitals": vitals is not None,
    }