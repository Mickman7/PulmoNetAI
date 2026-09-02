"""
Note: text extraction from uploaded TXT/PDF notes files happens in the
frontend before this call -- the API always receives plain `notes` text
plus numeric wbc/crp. `vitals`, if provided, arrives as a JSON string
(multipart forms can't carry nested arrays directly) representing a
24-row list of hourly readings, one column per vitals channel.
"""

import json
import logging
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from PIL import Image

from ...agent.utils import summarize_spatial_focus
from ...database.db import get_db
from ...database import crud
from ...models import inference
from .. import schema

router = APIRouter(prefix="/predict", tags=["predict"])
logger = logging.getLogger(__name__)

STORAGE_DIR = "backend/storage/patient_images"


@router.post("/{patient_id}", response_model=schema.PredictionOut)
def run_prediction(
    patient_id: int,
    image: UploadFile = File(...),
    notes: str = Form(...),
    wbc: float = Form(...),
    crp: float = Form(...),
    vitals: str | None = Form(None),
    db: Session = Depends(get_db),
):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    parsed_vitals = None
    if vitals:
        try:
            parsed_vitals = json.loads(vitals)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="vitals must be valid JSON (24-row list of lists)")

    patient_dir = os.path.join(STORAGE_DIR, str(patient_id))
    os.makedirs(patient_dir, exist_ok=True)
    image_path = os.path.join(patient_dir, f"{uuid.uuid4().hex}.jpg")

    pil_image = Image.open(image.file).convert("RGB")
    pil_image.save(image_path)

    result = inference.predict(image_path, notes, wbc, crp, vitals=parsed_vitals)

    # Best-effort: a Grad-CAM failure shouldn't block the prediction itself
    # from saving -- attention_summary/gradcam_base64 just stay unset.
    # attention_summary is derived from the Grad-CAM heatmap specifically
    # (not the fusion module's attn_weights) -- the heatmap is the only
    # tensor left that carries real per-patch spatial information since the
    # fusion redesign pools the image into a single token before a 4-way
    # [image, text, labs, vitals] self-attention (see agent/utils.py).
    gradcam_base64 = None
    attention_summary = None
    try:
        gradcam_result = inference.generate_gradcam_overlay(
            image_path, notes, wbc, crp, vitals=parsed_vitals
        )
        gradcam_base64 = gradcam_result["gradcam_base64"]
        attention_summary = summarize_spatial_focus(gradcam_result["heatmap"])
    except Exception:
        logger.exception("Grad-CAM generation failed for prediction on patient %s", patient_id)

    prediction = crud.create_prediction(db, patient_id, {
        "image_path": image_path,
        "notes": notes,
        "wbc": wbc,
        "crp": crp,
        "probability": result["probability"],
        "label": result["label"],
        "attention_summary": attention_summary,
    })
    prediction.gradcam_base64 = gradcam_base64

    return prediction