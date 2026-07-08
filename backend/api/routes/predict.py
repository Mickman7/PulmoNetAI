"""
Note: text extraction from uploaded TXT/PDF notes files happens in the
frontend (see pulmo_app / Streamlit pages) before this call -- the API
always receives plain `notes` text plus numeric wbc/crp. This keeps the
API contract simple and framework-agnostic.
"""

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

STORAGE_DIR = "backend/storage/patient_images"


@router.post("/{patient_id}", response_model=schema.PredictionOut)
def run_prediction(
    patient_id: int,
    image: UploadFile = File(...),
    notes: str = Form(...),
    wbc: float = Form(...),
    crp: float = Form(...),
    db: Session = Depends(get_db),
):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Save the image to disk first (path stored in DB, not the bytes themselves)
    patient_dir = os.path.join(STORAGE_DIR, str(patient_id))
    os.makedirs(patient_dir, exist_ok=True)
    image_path = os.path.join(patient_dir, f"{uuid.uuid4().hex}.jpg")

    pil_image = Image.open(image.file).convert("RGB")
    pil_image.save(image_path)

    result = inference.predict(image_path, notes, wbc, crp)

    prediction = crud.create_prediction(db, patient_id, {
        "image_path": image_path,
        "notes": notes,
        "wbc": wbc,
        "crp": crp,
        "probability": result["probability"],
        "label": result["label"],
    })
    return prediction
