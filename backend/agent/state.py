from typing import TypedDict, Optional, List


class PredictionRecord(TypedDict):
    """One selected past prediction — the agent may reason over several of these."""
    created_at: str
    probability: float
    label: str
    attention_focus: str
    notes: str
    wbc: Optional[float]
    crp: Optional[float]
    vitals_summary: Optional[str]


class AgentState(TypedDict):
    patient_summary: str              # structured medical history, rendered as text
    records: List[PredictionRecord]   # clinician-selected predictions, oldest -> newest
    analysis: Optional[str]
    reasoning: Optional[str]
    report: Optional[str]