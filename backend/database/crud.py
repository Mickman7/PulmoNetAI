"""
CRUD functions. Kept as plain functions (not a class) so both the FastAPI
routes and any script/notebook can import and call them directly.
"""

from sqlalchemy.orm import Session

from . import models


class PatientHasRecordsError(Exception):
    """Raised when trying to delete a patient who still has predictions or reports."""
    pass


# ---------------------------------------------------------------------------
# Patient CRUD
# ---------------------------------------------------------------------------
def create_patient(db: Session, data: dict) -> models.Patient:
    patient = models.Patient(**data)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def get_patient(db: Session, patient_id: int) -> models.Patient | None:
    return db.query(models.Patient).filter(models.Patient.id == patient_id).first()


def search_patients(db: Session, query: str) -> list[models.Patient]:
    # simple case-insensitive partial match on name
    return db.query(models.Patient).filter(models.Patient.name.ilike(f"%{query}%")).all()


def update_patient(db: Session, patient_id: int, data: dict) -> models.Patient | None:
    patient = get_patient(db, patient_id)
    if not patient:
        return None
    for key, value in data.items():
        setattr(patient, key, value)
    db.commit()
    db.refresh(patient)
    return patient


def delete_patient(db: Session, patient_id: int) -> None:
    patient = get_patient(db, patient_id)
    if not patient:
        return

    # Blocked delete: refuse if the patient has any prediction or report history.
    # This preserves the clinical audit trail rather than silently losing records.
    has_predictions = db.query(models.Prediction).filter(
        models.Prediction.patient_id == patient_id
    ).first()
    has_reports = db.query(models.Report).filter(
        models.Report.patient_id == patient_id
    ).first()

    if has_predictions or has_reports:
        raise PatientHasRecordsError(
            f"Cannot delete patient {patient_id}: existing predictions/reports must be removed first."
        )

    db.delete(patient)
    db.commit()

def get_all_patients(db: Session) -> list[models.Patient]:
    return db.query(models.Patient).order_by(models.Patient.name).all()


# ---------------------------------------------------------------------------
# Prediction CRUD
# ---------------------------------------------------------------------------
def create_prediction(db: Session, patient_id: int, data: dict) -> models.Prediction:
    prediction = models.Prediction(patient_id=patient_id, **data)
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_predictions_for_patient(db: Session, patient_id: int) -> list[models.Prediction]:
    return db.query(models.Prediction).filter(
        models.Prediction.patient_id == patient_id
    ).order_by(models.Prediction.created_at.desc()).all()


def get_predictions_by_ids(db: Session, prediction_ids: list[int]) -> list[models.Prediction]:
    return db.query(models.Prediction).filter(models.Prediction.id.in_(prediction_ids)).all()


def delete_prediction(db: Session, prediction_id: int) -> None:
    prediction = db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()
    if prediction:
        db.delete(prediction)
        db.commit()


# ---------------------------------------------------------------------------
# Report CRUD
# ---------------------------------------------------------------------------
def create_report(db: Session, patient_id: int, data: dict) -> models.Report:
    report = models.Report(patient_id=patient_id, **data)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_reports_for_patient(db: Session, patient_id: int) -> list[models.Report]:
    return db.query(models.Report).filter(
        models.Report.patient_id == patient_id
    ).order_by(models.Report.created_at.desc()).all()


def delete_report(db: Session, report_id: int) -> None:
    report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if report:
        db.delete(report)
        db.commit()