"""Thin wrapper around the Pulmo API. Pages import from here rather than
calling `requests` directly, so the base URL only needs to change in one place."""

import requests

API_BASE_URL = "http://localhost:8000"  # <-- update if your API runs elsewhere


def search_patients(query: str) -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/patients/search", params={"q": query})
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


def run_prediction(patient_id: int, image_bytes: bytes, image_name: str, notes: str, wbc: float, crp: float) -> dict:
    files = {"image": (image_name, image_bytes, "image/jpeg")}
    data = {"notes": notes, "wbc": wbc, "crp": crp}
    r = requests.post(f"{API_BASE_URL}/predict/{patient_id}", files=files, data=data)
    r.raise_for_status()
    return r.json()


def run_agent(patient_id: int, prediction_ids: list[int]) -> dict:
    payload = {"patient_id": patient_id, "prediction_ids": prediction_ids}
    r = requests.post(f"{API_BASE_URL}/agent/run", json=payload)
    r.raise_for_status()
    return r.json()

def list_patients() -> list[dict]:
    r = requests.get(f"{API_BASE_URL}/patients")
    r.raise_for_status()
    return r.json()
