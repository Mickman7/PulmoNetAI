"""
Vitals CSV template + parser for the Predict page.
Column order must exactly match the 8-channel order the model was trained
on (see backend/multimodal_dataset_pipeline.py's FEATURE_COLS): HR, O2Sat,
Temp, Resp, SBP, MAP, WBC, FiO2. If that training order ever changes, this
file's VITALS_COLUMNS list needs to be updated to match.
"""

import io

import pandas as pd

HOURS = 24
VITALS_COLUMNS = [
    "heart_rate", "spo2", "temperature", "respiratory_rate",
    "systolic_bp", "map", "wbc", "fio2",
]

# Neutral defaults for the template
_TEMPLATE_DEFAULTS = {
    "heart_rate": 90.0,
    "spo2": 93.0,
    "temperature": 37.0,
    "respiratory_rate": 21.0,
    "systolic_bp": 120.0,
    "map": 80.0,
    "wbc": 8.0,
    "fio2": 21.0,  # 21% = room air, the normal default when a patient isn't on supplemental oxygen
}


def generate_vitals_template() -> bytes:
    """Returns a CSV (as bytes) with 24 hourly rows and one column per
    vitals channel, pre-filled with neutral defaults -- for the clinician
    to download, edit, and re-upload."""
    df = pd.DataFrame({
        "hour": range(HOURS),
        **{col: [_TEMPLATE_DEFAULTS[col]] * HOURS for col in VITALS_COLUMNS},
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def parse_vitals_csv(uploaded_file) -> list[list[float]]:
    """Validates and converts an uploaded CSV into the [24, 8] nested list
    the API expects. Raises ValueError with a clear message on any mismatch."""
    df = pd.read_csv(uploaded_file)

    if "hour" in df.columns:
        df = df.drop(columns=["hour"])

    if len(df) != HOURS:
        raise ValueError(f"Expected {HOURS} rows (one per hour), got {len(df)}.")

    missing = [c for c in VITALS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}. "
                          f"Expected columns: {VITALS_COLUMNS}")

    df = df[VITALS_COLUMNS]  # enforce correct column order

    if df.isnull().any().any():
        raise ValueError("CSV contains empty/missing values -- every hour must have a value for every column.")

    return df.astype(float).values.tolist()