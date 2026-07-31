from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...database.db import get_db
from ...database import crud
from .. import schema

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("", response_model=schema.PatientOut)
def create_patient(payload: schema.PatientCreate, db: Session = Depends(get_db)):
    return crud.create_patient(db, payload.model_dump())


@router.get("", response_model=list[schema.PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return crud.get_all_patients(db)


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
    updated = crud.update_patient(db, patient_id, payload.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Patient not found")
    return updated


@router.delete("/{patient_id}")
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    try:
        crud.delete_patient(db, patient_id)
    except crud.PatientHasRecordsError as e:
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


@router.post("/{patient_id}/encounters", response_model=schema.EncounterOut)
def add_encounter(patient_id: int, payload: schema.EncounterCreate, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    data = payload.model_dump(exclude_unset=True)
    return crud.create_encounter(db, patient_id, data)


@router.get("/{patient_id}/encounters", response_model=list[schema.EncounterOut])
def get_encounters(patient_id: int, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return crud.get_encounters_for_patient(db, patient_id)


@router.post("/{patient_id}/appointments", response_model=schema.AppointmentOut)
def add_appointment(patient_id: int, payload: schema.AppointmentCreate, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return crud.create_appointment(db, patient_id, payload.model_dump())


@router.get("/{patient_id}/appointments", response_model=list[schema.AppointmentOut])
def get_appointments(patient_id: int, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return crud.get_appointments_for_patient(db, patient_id)


@router.post("/{patient_id}/treatments", response_model=schema.TreatmentOut)
def add_treatment(patient_id: int, payload: schema.TreatmentCreate, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return crud.create_treatment(db, patient_id, payload.model_dump())


@router.get("/{patient_id}/treatments", response_model=list[schema.TreatmentOut])
def get_treatments(patient_id: int, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return crud.get_treatments_for_patient(db, patient_id)