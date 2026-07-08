from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...database.db import get_db
from ...database import crud
from .. import schema

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("", response_model=schema.PatientOut)
def create_patient(payload: schema.PatientCreate, db: Session = Depends(get_db)):
    return crud.create_patient(db, payload.model_dump())


@router.get("/search", response_model=list[schema.PatientOut])
def search_patients(q: str, db: Session = Depends(get_db)):
    return crud.search_patients(db, q)


@router.get("/{patient_id}", response_model=schema.PatientOut)
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}", response_model=schema.PatientOut)
def update_patient(patient_id: int, payload: schema.PatientUpdate, db: Session = Depends(get_db)):
    # exclude_unset -> only overwrite fields the clinician actually sent
    updated = crud.update_patient(db, patient_id, payload.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Patient not found")
    return updated


@router.delete("/{patient_id}")
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    try:
        crud.delete_patient(db, patient_id)
    except crud.PatientHasRecordsError as e:
        # 409 Conflict -- the resource can't be deleted in its current state
        raise HTTPException(status_code=409, detail=str(e))
    return {"detail": "Patient deleted"}


@router.get("/{patient_id}/predictions", response_model=list[schema.PredictionOut])
def get_patient_predictions(patient_id: int, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return crud.get_predictions_for_patient(db, patient_id)


@router.get("/{patient_id}/reports", response_model=list[schema.ReportOut])
def get_patient_reports(patient_id: int, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return crud.get_reports_for_patient(db, patient_id)
