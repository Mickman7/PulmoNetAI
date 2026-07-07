"""
ORM table definitions.

Patient  -- one row per patient, structured medical history
Prediction -- one row per model run (image + notes + labs -> probability), many per patient
Report   -- one row per agent run (analysis/reasoning/report), built from selected Predictions
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Text, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from .db import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)  # indexed for fast search
    date_of_birth = Column(String, nullable=True)
    sex = Column(String, nullable=True)
    contact_info = Column(String, nullable=True)

    # Structured medical history — stored as JSON so each field can hold a list
    allergies = Column(JSON, default=list)
    chronic_conditions = Column(JSON, default=list)
    current_medications = Column(JSON, default=list)
    past_surgeries = Column(JSON, default=list)
    smoking_status = Column(String, nullable=True)  # "never" / "former" / "current"
    family_history = Column(Text, nullable=True)     # free text, too varied to structure

    created_at = Column(DateTime, default=datetime.utcnow)

    # cascade left off deliberately -- deletion is blocked in crud.py if these exist
    predictions = relationship("Prediction", back_populates="patient")
    reports = relationship("Report", back_populates="patient")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    image_path = Column(String, nullable=False)  # file on disk, e.g. storage/patient_images/3/7.jpg
    notes = Column(Text, nullable=True)
    wbc = Column(Float, nullable=True)
    crp = Column(Float, nullable=True)

    probability = Column(Float, nullable=False)
    label = Column(String, nullable=False)  # "Pneumonia" / "Normal"

    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    # which Prediction rows the clinician chose to feed the agent, e.g. [5, 6]
    source_prediction_ids = Column(JSON, default=list)

    analysis = Column(Text, nullable=True)
    reasoning = Column(Text, nullable=True)
    report_text = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="reports")