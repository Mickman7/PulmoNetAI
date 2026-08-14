from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Text, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from .db import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    date_of_birth = Column(String, nullable=True)
    sex = Column(String, nullable=True)
    contact_info = Column(String, nullable=True)

    allergies = Column(JSON, default=list)
    chronic_conditions = Column(JSON, default=list)
    current_medications = Column(JSON, default=list)
    past_surgeries = Column(JSON, default=list)
    smoking_status = Column(String, nullable=True)
    family_history = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="patient")
    reports = relationship("Report", back_populates="patient")
    encounters = relationship("ClinicalEncounter", back_populates="patient")
    appointments = relationship("Appointment", back_populates="patient")
    treatments = relationship("Treatment", back_populates="patient")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    image_path = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    wbc = Column(Float, nullable=True)
    crp = Column(Float, nullable=True)
    attention_summary = Column(String, nullable=True)

    probability = Column(Float, nullable=False)
    label = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    source_prediction_ids = Column(JSON, default=list)

    analysis = Column(Text, nullable=True)
    reasoning = Column(Text, nullable=True)
    report_text = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="reports")


class ClinicalEncounter(Base):
    """Dated clinical intake -- vitals, physical exam, and diagnostic extras
    that the report template references but aren't produced by the model
    (BP, temperature, auscultation/percussion, microbiology, ABG)."""
    __tablename__ = "clinical_encounters"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    encounter_date = Column(DateTime, default=datetime.utcnow)

    # Vitals
    heart_rate = Column(Float, nullable=True)
    blood_pressure = Column(String, nullable=True)  # e.g. "120/80"
    respiratory_rate = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    spo2 = Column(Float, nullable=True)

    # Physical exam
    general_appearance = Column(Text, nullable=True)
    chest_auscultation = Column(Text, nullable=True)
    percussion = Column(Text, nullable=True)

    # Diagnostic extras
    microbiology = Column(Text, nullable=True)
    abg = Column(Text, nullable=True)  # arterial blood gas, free text (pH/pO2/pCO2)

    notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="encounters")


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    appointment_date = Column(DateTime, nullable=False)
    reason = Column(String, nullable=True)
    status = Column(String, nullable=True)  # scheduled / completed / cancelled
    notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="appointments")


class Treatment(Base):
    __tablename__ = "treatments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    start_date = Column(DateTime, nullable=False)
    description = Column(String, nullable=False)  # e.g. "Amoxicillin"
    dosage = Column(String, nullable=True)
    route = Column(String, nullable=True)  # oral / IV / etc.
    duration = Column(String, nullable=True)
    notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="treatments")