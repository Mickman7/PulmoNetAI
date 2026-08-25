import base64

import streamlit as st

from api_client import search_patients, run_prediction
from file_parsing import extract_text, parse_labs
from vitals_utils import generate_vitals_template, parse_vitals_csv

st.set_page_config(layout="wide")

st.title("Predict")

# --- 1. Search and select a patient ---------------------------------------
st.header("1. Select Patient")
query = st.text_input("Search patient by name")

selected_patient = None
if query:
    results = search_patients(query)
    if not results:
        st.warning("No matching patients found.")
    else:
        options = {f"{p['name']} (ID: {p['id']})": p for p in results}
        choice = st.selectbox("Matching patients", list(options.keys()))
        selected_patient = options[choice]

if selected_patient:
    st.success(f"Selected: {selected_patient['name']}")

# --- 2. X-ray upload ---------------------------------------------------
st.header("2. Chest X-ray")
image_file = st.file_uploader("Upload a radiograph (PNG/JPEG)", type=["png", "jpg", "jpeg"])
if image_file:
    st.image(image_file, caption="Uploaded radiograph", width=300)

# --- 3. Notes & labs -----------------------------------------------------
st.header("3. Clinical Notes & Lab Results")
manual_entry = st.checkbox("Enter manually instead of uploading a file")

notes, wbc, crp = "", None, None

if manual_entry:
    notes = st.text_area("Clinical notes")
    col1, col2 = st.columns(2)
    with col1:
        wbc = st.number_input("WBC Count (x10^9/L)", min_value=0.0, step=0.1)
    with col2:
        crp = st.number_input("CRP Level (mg/L)", min_value=0.0, step=0.1)
else:
    notes_file = st.file_uploader("Upload notes/labs (TXT or PDF)", type=["txt", "pdf"])
    if notes_file:
        raw_text = extract_text(notes_file)
        parsed_wbc, parsed_crp = parse_labs(raw_text)

        notes = st.text_area("Extracted notes (edit if needed)", value=raw_text, height=150)
        col1, col2 = st.columns(2)
        with col1:
            wbc = st.number_input("WBC Count (x10^9/L)", value=parsed_wbc or 0.0, step=0.1)
        with col2:
            crp = st.number_input("CRP Level (mg/L)", value=parsed_crp or 0.0, step=0.1)

        if parsed_wbc is None or parsed_crp is None:
            st.info("Couldn't auto-detect WBC/CRP from the file -- please check the values above.")

# --- 4. Vitals (optional) ---------------------------------------------------
st.header("4. Vitals (Optional)")
st.caption(
    "24-hour hourly readings. If not provided, the model treats vitals as "
    "unavailable rather than guessing -- this doesn't hurt the prediction, "
    "it just means that signal isn't used."
)

col_a, col_b = st.columns([1, 2])
with col_a:
    st.download_button(
        "Download CSV Template",
        data=generate_vitals_template(),
        file_name="vitals_template.csv",
        mime="text/csv",
    )

vitals_file = st.file_uploader("Upload vitals CSV", type=["csv"])
vitals_data = None
if vitals_file:
    try:
        vitals_data = parse_vitals_csv(vitals_file)
        st.success(f"Vitals loaded: {len(vitals_data)} hourly readings.")
    except ValueError as e:
        st.error(f"Invalid vitals CSV: {e}")

# --- 5. Predict ----------------------------------------------------------
st.header("5. Prediction")
if st.button("Predict", type="primary"):
    if selected_patient is None:
        st.error("Please select a patient.")
    elif image_file is None:
        st.error("Please upload a radiograph image.")
    elif not notes.strip():
        st.error("Please provide clinical notes.")
    else:
        with st.spinner("Running prediction..."):
            try:
                result = run_prediction(
                    patient_id=selected_patient["id"],
                    image_bytes=image_file.getvalue(),
                    image_name=image_file.name,
                    notes=notes,
                    wbc=wbc,
                    crp=crp,
                    vitals=vitals_data,
                )
                st.subheader("Result")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", result["label"])
                col2.metric("Probability", f"{result['probability']:.1%}")
                st.progress(result["probability"])

                if result.get("gradcam_base64"):
                    st.subheader("Grad-CAM")
                    st.caption("Highlighted regions most influenced the model's prediction.")
                    st.image(
                        base64.b64decode(result["gradcam_base64"]),
                        caption="Grad-CAM overlay",
                        width=300,
                    )
                else:
                    st.caption("Grad-CAM heatmap unavailable for this prediction.")

                if vitals_data is not None:
                    st.caption("Prediction used your uploaded vitals.")
                else:
                    st.caption("Prediction did not include vitals (none provided).")

                st.caption("Saved to patient's record.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.info("Search for a patient above to begin.")