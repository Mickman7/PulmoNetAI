from sqlalchemy.orm import Session

from . import models


class PatientHasRecordsError(Exception):
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


def get_all_patients(db: Session) -> list[models.Patient]:
    return db.query(models.Patient).order_by(models.Patient.name).all()


def search_patients(db: Session, query: str) -> list[models.Patient]:
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

    has_predictions = db.query(models.Prediction).filter(models.Prediction.patient_id == patient_id).first()
    has_reports = db.query(models.Report).filter(models.Report.patient_id == patient_id).first()
    has_encounters = db.query(models.ClinicalEncounter).filter(models.ClinicalEncounter.patient_id == patient_id).first()
    has_appointments = db.query(models.Appointment).filter(models.Appointment.patient_id == patient_id).first()
    has_treatments = db.query(models.Treatment).filter(models.Treatment.patient_id == patient_id).first()

    if any([has_predictions, has_reports, has_encounters, has_appointments, has_treatments]):
        raise PatientHasRecordsError(
            f"Cannot delete patient {patient_id}: existing clinical records must be removed first."
        )

    db.delete(patient)
    db.commit()


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


# ---------------------------------------------------------------------------
# ClinicalEncounter CRUD
# ---------------------------------------------------------------------------
def create_encounter(db: Session, patient_id: int, data: dict) -> models.ClinicalEncounter:
    encounter = models.ClinicalEncounter(patient_id=patient_id, **data)
    db.add(encounter)
    db.commit()
    db.refresh(encounter)
    return encounter


def get_encounters_for_patient(db: Session, patient_id: int) -> list[models.ClinicalEncounter]:
    return db.query(models.ClinicalEncounter).filter(
        models.ClinicalEncounter.patient_id == patient_id
    ).order_by(models.ClinicalEncounter.encounter_date.desc()).all()


# ---------------------------------------------------------------------------
# Appointment CRUD
# ---------------------------------------------------------------------------
def create_appointment(db: Session, patient_id: int, data: dict) -> models.Appointment:
    appointment = models.Appointment(patient_id=patient_id, **data)
    db.add(appointment)
    db.commit()
    db.refresh(appointment)
    return appointment


def get_appointments_for_patient(db: Session, patient_id: int) -> list[models.Appointment]:
    return db.query(models.Appointment).filter(
        models.Appointment.patient_id == patient_id
    ).order_by(models.Appointment.appointment_date.desc()).all()


# ---------------------------------------------------------------------------
# Treatment CRUD
# ---------------------------------------------------------------------------
def create_treatment(db: Session, patient_id: int, data: dict) -> models.Treatment:
    treatment = models.Treatment(patient_id=patient_id, **data)
    db.add(treatment)
    db.commit()
    db.refresh(treatment)
    return treatment


def get_treatments_for_patient(db: Session, patient_id: int) -> list[models.Treatment]:
    return db.query(models.Treatment).filter(
        models.Treatment.patient_id == patient_id
    ).order_by(models.Treatment.start_date.desc()).all()