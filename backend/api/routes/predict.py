import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from PIL import Image
from codecarbon import EmissionsTracker

from ...database.db import get_db
from ...database import crud
from ...models import inference
from .. import schema
from backend.agent.utils import summarize_attention


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

    # Use absolute paths to eliminate workspace profile conflicts
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    evaluation_results_dir = os.path.join(base_dir, "evaluation_results")
    os.makedirs(evaluation_results_dir, exist_ok=True)

    # 1. Generate absolute pathing for the execution workspace
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    evaluation_results_dir = os.path.join(base_dir, "evaluation_results")
    os.makedirs(evaluation_results_dir, exist_ok=True)

    # 2. Write a clean configuration file directly into the active working directory
    config_path = os.path.join(os.getcwd(), ".codecarbon.config")
    with open(config_path, "w") as f:
        f.write("[codecarbon]\n")
        f.write(f"project_name = pulmonet_live_inference\n")
        f.write("save_to_file = true\n")
        f.write(f"output_dir = {evaluation_results_dir}\n")
        f.write("log_level = warning\n")

    tracker = EmissionsTracker()
    
    
    tracker.start()
    try:
        # 1. Run the multimodal forward pass
        result = inference.predict(image_path, notes, wbc, crp)
    finally:
        tracker.stop()


    # 2. Extract attention summary text to persist in the database
    attention_string_label = summarize_attention(result["attn_weights"])

    # 3. Create the database row with the saved attention summary text
    prediction = crud.create_prediction(db, patient_id, {
        "image_path": image_path,
        "notes": notes,
        "wbc": wbc,
        "crp": crp,
        "probability": result["probability"],
        "label": result["label"],
        "attention_summary": attention_string_label,
    })
    return prediction