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

    prediction = crud.create_prediction(db, patient_id, {
        "image_path": image_path,
        "notes": notes,
        "wbc": wbc,
        "crp": crp,
        "probability": result["probability"],
        "label": result["label"],
    })

    # Best-effort: the prediction itself already succeeded and is saved, so a
    # Grad-CAM failure shouldn't fail the whole request -- just ship without it.
    try:
        prediction.gradcam_base64 = inference.generate_gradcam_overlay(
            image_path, notes, wbc, crp, vitals=parsed_vitals
        )
    except Exception:
        logger.exception("Grad-CAM generation failed for prediction %s", prediction.id)
        prediction.gradcam_base64 = None

    return prediction