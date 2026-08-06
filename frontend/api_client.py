"""Thin wrapper around the Pulmo API. Pages import from here rather than
calling `requests` directly, so the base URL only needs to change in one place."""

import json

import requests

API_BASE_URL = "http://localhost:8000"  # <-- update if your API runs elsewhere


def list_patients() -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/patients")
    r.raise_for_status()
    return r.json()


def search_patients(query: str) -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/patients/search", params={"q": query})
    r.raise_for_status()
    return r.json()


def get_patient(patient_id: int) -> dict:
    r = requests.get(f"{API_BASE_URL}/patients/{patient_id}")
    r.raise_for_status()
    return r.json()


def update_patient(patient_id: int, payload: dict) -> dict:
    r = requests.put(f"{API_BASE_URL}/patients/{patient_id}", json=payload)
    r.raise_for_status()
    return r.json()


def create_patient(payload: dict) -> dict:
    r = requests.post(f"{API_BASE_URL}/patients", json=payload)
    r.raise_for_status()
    return r.json()


def get_patient_predictions(patient_id: int) -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/patients/{patient_id}/predictions")
    r.raise_for_status()
    return r.json()


def run_prediction(patient_id: int, image_bytes: bytes, image_name: str, notes: str,
                    wbc: float, crp: float, vitals: list[list[float]] | None = None) -> dict:
    """vitals, if provided, should be a 24-row list of lists (one row per hour,
    one column per vitals channel), matching the model's expected shape."""
    files = {"image": (image_name, image_bytes, "image/jpeg")}
    data = {"notes": notes, "wbc": wbc, "crp": crp}
    if vitals is not None:
        data["vitals"] = json.dumps(vitals)  # sent as a JSON string field alongside the multipart upload
    r = requests.post(f"{API_BASE_URL}/predict/{patient_id}", files=files, data=data)
    r.raise_for_status()
    return r.json()


def run_agent(patient_id: int, prediction_ids: list[int]) -> dict:
    payload = {"patient_id": patient_id, "prediction_ids": prediction_ids}
    r = requests.post(f"{API_BASE_URL}/agent/run", json=payload)
    r.raise_for_status()
    return r.json()


# --- Clinical encounters (vitals, physical exam, microbiology, ABG) --------
def get_encounters(patient_id: int) -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/patients/{patient_id}/encounters")
    r.raise_for_status()
    return r.json()


def add_encounter(patient_id: int, payload: dict) -> dict:
    r = requests.post(f"{API_BASE_URL}/patients/{patient_id}/encounters", json=payload)
    r.raise_for_status()
    return r.json()


# --- Appointments -----------------------------------------------------------
def get_appointments(patient_id: int) -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/patients/{patient_id}/appointments")
    r.raise_for_status()
    return r.json()


def add_appointment(patient_id: int, payload: dict) -> dict:
    r = requests.post(f"{API_BASE_URL}/patients/{patient_id}/appointments", json=payload)
    r.raise_for_status()
    return r.json()


def update_appointment(patient_id: int, appointment_id: int, payload: dict) -> dict:
    r = requests.put(f"{API_BASE_URL}/patients/{patient_id}/appointments/{appointment_id}", json=payload)
    r.raise_for_status()
    return r.json()


def delete_appointment(patient_id: int, appointment_id: int) -> None:
    r = requests.delete(f"{API_BASE_URL}/patients/{patient_id}/appointments/{appointment_id}")
    r.raise_for_status()


# --- Treatments ---------------------------------------------------------------
def get_treatments(patient_id: int) -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/patients/{patient_id}/treatments")
    r.raise_for_status()
    return r.json()


def add_treatment(patient_id: int, payload: dict) -> dict:
    r = requests.post(f"{API_BASE_URL}/patients/{patient_id}/treatments", json=payload)
    r.raise_for_status()
    return r.json()


def update_treatment(patient_id: int, treatment_id: int, payload: dict) -> dict:
    r = requests.put(f"{API_BASE_URL}/patients/{patient_id}/treatments/{treatment_id}", json=payload)
    r.raise_for_status()
    return r.json()


def delete_treatment(patient_id: int, treatment_id: int) -> None:
    r = requests.delete(f"{API_BASE_URL}/patients/{patient_id}/treatments/{treatment_id}")
    r.raise_for_status()