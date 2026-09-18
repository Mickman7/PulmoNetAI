"""
Note on attention weights: attn_weights are only computed at inference time
and are not persisted to the DB (they're a large tensor, not something you'd
want stored per-row in SQLite). For past predictions, we mark attention_focus
as unavailable rather than silently fabricating it -- the analysis node still
has the probability, label, and lab/notes trend to reason over.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...database.db import get_db
from ...database import crud
from ...agent.graph import agent
from ...agent.utils import (
    build_patient_summary,
    find_nearest_encounter,
    format_vitals_summary,
    format_vitals_timeseries_summary,
)
from .. import schema

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run", response_model=schema.ReportOut)
def run_agent(payload: schema.AgentRunRequest, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, payload.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    predictions = crud.get_predictions_by_ids(db, payload.prediction_ids)
    if not predictions:
        raise HTTPException(status_code=400, detail="No matching predictions found for the given IDs")

    # oldest -> newest, so the analysis node can reason about trends in order
    predictions = sorted(predictions, key=lambda p: p.created_at)

    encounters = crud.get_encounters_for_patient(db, payload.patient_id)

    def _vitals_summary(p) -> str | None:
        # Two independent vitals sources may exist for a given prediction: a
        # clinician-recorded encounter snapshot, and/or the 24h monitor feed
        # uploaded alongside the image on the Predict page. Surface both when
        # present rather than picking one.
        parts = [
            format_vitals_summary(find_nearest_encounter(encounters, p.created_at)),
            format_vitals_timeseries_summary(p.vitals),
        ]
        parts = [part for part in parts if part]
        return "; ".join(parts) if parts else None

    records = [
        {
            "created_at": p.created_at.isoformat(),
            "probability": p.probability,
            "label": p.label,
            # Read the saved text description instead of hardcoding "Not available"
            "attention_focus": p.attention_summary if p.attention_summary else "spread broadly across the image (suggests a diffuse or less certain finding)",
            "notes": p.notes or "",
            "wbc": p.wbc,
            "crp": p.crp,
            "vitals_summary": _vitals_summary(p),
        }
        for p in predictions
    ]

    initial_state = {
        "patient_summary": build_patient_summary(patient),
        "records": records,
        "rag_mode": payload.rag_mode,
        "analysis": None,
        "reasoning": None,
        "report": None,
    }

    final_state = agent.invoke(initial_state)

    report = crud.create_report(db, payload.patient_id, {
        "source_prediction_ids": payload.prediction_ids,
        "analysis": final_state["analysis"],
        "reasoning": final_state["reasoning"],
        "report_text": final_state["report"],
    })
    return report