"""
Vitals CSV template + parser for the Predict page.

Only 3 of the 8 channels are confirmed (heart_rate, respiratory_rate, spo2).
The remaining 5 are placeholder columns (vital_4..vital_8) until their real
meaning is decided -- rename these columns here AND in synthetic_vitals.py /
multimodal_system.py's VITALS_CHANNELS constant together when that happens.
"""

import io

import pandas as pd

HOURS = 24
VITALS_COLUMNS = [
    "heart_rate", "respiratory_rate", "spo2",
    "vital_4", "vital_5", "vital_6", "vital_7", "vital_8",  # placeholders -- rename once decided
]

# Neutral defaults for the template: midpoint values for the known 3,
# 0.0 placeholders for the unconfirmed 5 (edit once their ranges are known)
_TEMPLATE_DEFAULTS = {
    "heart_rate": 90.0,
    "respiratory_rate": 21.0,
    "spo2": 93.0,
    "vital_4": 0.0, "vital_5": 0.0, "vital_6": 0.0, "vital_7": 0.0, "vital_8": 0.0,
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