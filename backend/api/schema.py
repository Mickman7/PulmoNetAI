from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------
class PatientCreate(BaseModel):
    name: str
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None
    contact_info: Optional[str] = None
    allergies: list[str] = []
    chronic_conditions: list[str] = []
    current_medications: list[str] = []
    past_surgeries: list[str] = []
    smoking_status: Optional[str] = None
    family_history: Optional[str] = None
 
 
class PatientUpdate(BaseModel):
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None
    contact_info: Optional[str] = None
    allergies: Optional[list[str]] = None
    chronic_conditions: Optional[list[str]] = None
    current_medications: Optional[list[str]] = None
    past_surgeries: Optional[list[str]] = None
    smoking_status: Optional[str] = None
    family_history: Optional[str] = None
 
 
class PatientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
 
    id: int
    name: str
    date_of_birth: Optional[str]
    sex: Optional[str]
    contact_info: Optional[str]
    allergies: list[str]
    chronic_conditions: list[str]
    current_medications: list[str]
    past_surgeries: list[str]
    smoking_status: Optional[str]
    family_history: Optional[str]
    created_at: datetime

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class PredictionOut(BaseModel):
    id: int
    patient_id: int
    image_path: str
    notes: Optional[str]
    wbc: Optional[float]
    crp: Optional[float]
    probability: float
    label: str
    # --- ADD THIS STRING TO MATCH PIPELINE PERSISTENCE ---
    attention_summary: Optional[str] = None
    created_at: datetime
    # Not a DB column -- set as a plain attribute on the ORM object in the
    # predict route before serialization, so it rides along in the response
    # without needing a migration.
    gradcam_base64: Optional[str] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Agent / Report
# ---------------------------------------------------------------------------
class AgentRunRequest(BaseModel):
    patient_id: int
    prediction_ids: List[int]  # which past predictions the clinician selected
    rag_mode: Literal["local", "pubmed", "hybrid"] = "hybrid"


class ReportOut(BaseModel):
    id: int
    patient_id: int
    source_prediction_ids: List[int]
    analysis: Optional[str]
    reasoning: Optional[str]
    report_text: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True



# ---------------------------------------------------------------------------
# ClinicalEncounter
# ---------------------------------------------------------------------------
class EncounterCreate(BaseModel):
    encounter_date: Optional[datetime] = None
    heart_rate: Optional[float] = None
    blood_pressure: Optional[str] = None
    respiratory_rate: Optional[float] = None
    temperature: Optional[float] = None
    spo2: Optional[float] = None
    general_appearance: Optional[str] = None
    chest_auscultation: Optional[str] = None
    percussion: Optional[str] = None
    microbiology: Optional[str] = None
    abg: Optional[str] = None
    notes: Optional[str] = None
 
 
class EncounterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
 
    id: int
    patient_id: int
    encounter_date: datetime
    heart_rate: Optional[float]
    blood_pressure: Optional[str]
    respiratory_rate: Optional[float]
    temperature: Optional[float]
    spo2: Optional[float]
    general_appearance: Optional[str]
    chest_auscultation: Optional[str]
    percussion: Optional[str]
    microbiology: Optional[str]
    abg: Optional[str]
    notes: Optional[str]
 
 
# ---------------------------------------------------------------------------
# Appointment
# ---------------------------------------------------------------------------
class AppointmentCreate(BaseModel):
    appointment_date: datetime
    reason: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
 
 
class AppointmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
 
    id: int
    patient_id: int
    appointment_date: datetime
    reason: Optional[str]
    status: Optional[str]
    notes: Optional[str]
 
 
# ---------------------------------------------------------------------------
# Treatment
# ---------------------------------------------------------------------------
class TreatmentCreate(BaseModel):
    start_date: datetime
    description: str
    dosage: Optional[str] = None
    route: Optional[str] = None
    duration: Optional[str] = None
    notes: Optional[str] = None
 
 
class TreatmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
 
    id: int
    patient_id: int
    start_date: datetime
    description: str
    dosage: Optional[str]
    route: Optional[str]
    duration: Optional[str]
    notes: Optional[str]