from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------
class PatientCreate(BaseModel):
    name: str
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None
    contact_info: Optional[str] = None
    allergies: List[str] = []
    chronic_conditions: List[str] = []
    current_medications: List[str] = []
    past_surgeries: List[str] = []
    smoking_status: Optional[str] = None
    family_history: Optional[str] = None


class PatientUpdate(PatientCreate):
    name: Optional[str] = None  # allow partial updates


class PatientOut(PatientCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


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

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Agent / Report
# ---------------------------------------------------------------------------
class AgentRunRequest(BaseModel):
    patient_id: int
    prediction_ids: List[int]  # which past predictions the clinician selected


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