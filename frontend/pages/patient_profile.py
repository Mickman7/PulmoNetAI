from datetime import date, datetime

import pandas as pd
import streamlit as st

from api_client import (
    search_patients, get_patient,
    get_patient_predictions,
    get_encounters, add_encounter,
    get_appointments, add_appointment,
    get_treatments, add_treatment,
)

st.title("Patient Profile")

# --- Resolve which patient to show -----------------------------------------
# Arrives via st.session_state if navigated from the Patients table; falls
# back to search here too, so this page also works as a direct entry point.
patient_id = st.session_state.get("selected_patient_id")

if not patient_id:
    query = st.text_input("Search patient by name")
    if query:
        results = search_patients(query)
        if not results:
            st.warning("No matching patients found.")
        else:
            options = {f"{p['name']} (ID: {p['id']})": p["id"] for p in results}
            choice = st.selectbox("Matching patients", list(options.keys()))
            patient_id = options[choice]

if not patient_id:
    st.info("Search for a patient above, or select one from the Patients page.")
    st.stop()

patient = get_patient(patient_id)

# --- Header -------------------------------------------------------------------
st.header(patient["name"])
col1, col2, col3 = st.columns(3)
col1.metric("Date of Birth", patient["date_of_birth"] or "—")
col2.metric("Sex", patient["sex"] or "—")
col3.metric("Contact", patient["contact_info"] or "—")

with st.expander("Medical History", expanded=True):
    st.write(f"**Allergies:** {', '.join(patient['allergies']) or 'None recorded'}")
    st.write(f"**Chronic conditions:** {', '.join(patient['chronic_conditions']) or 'None recorded'}")
    st.write(f"**Current medications:** {', '.join(patient['current_medications']) or 'None recorded'}")
    st.write(f"**Past surgeries:** {', '.join(patient['past_surgeries']) or 'None recorded'}")
    st.write(f"**Smoking status:** {patient['smoking_status'] or 'Not recorded'}")
    st.write(f"**Family history:** {patient['family_history'] or 'Not recorded'}")

st.divider()

tab_encounters, tab_appointments, tab_treatments, tab_predictions = st.tabs(
    ["Clinical Encounters", "Appointments", "Treatments", "Predictions"]
)

# --- Clinical Encounters (vitals, physical exam, microbiology, ABG) --------
with tab_encounters:
    encounters = get_encounters(patient_id)
    if encounters:
        for e in encounters:
            with st.container(border=True):
                st.caption(e["encounter_date"][:16].replace("T", " "))
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("HR", e["heart_rate"] or "—")
                c2.metric("BP", e["blood_pressure"] or "—")
                c3.metric("RR", e["respiratory_rate"] or "—")
                c4.metric("Temp", e["temperature"] or "—")
                c5.metric("SpO2", e["spo2"] or "—")
                if e["general_appearance"]:
                    st.write(f"**General appearance:** {e['general_appearance']}")
                if e["chest_auscultation"]:
                    st.write(f"**Chest auscultation:** {e['chest_auscultation']}")
                if e["percussion"]:
                    st.write(f"**Percussion:** {e['percussion']}")
                if e["microbiology"]:
                    st.write(f"**Microbiology:** {e['microbiology']}")
                if e["abg"]:
                    st.write(f"**ABG:** {e['abg']}")
                if e["notes"]:
                    st.write(f"**Notes:** {e['notes']}")
    else:
        st.info("No clinical encounters recorded yet.")

    if "show_add_encounter" not in st.session_state:
        st.session_state.show_add_encounter = False
    if st.button("+ Add Clinical Encounter"):
        st.session_state.show_add_encounter = not st.session_state.show_add_encounter

    if st.session_state.show_add_encounter:
        with st.form("add_encounter_form"):
            st.markdown("**Vitals**")
            c1, c2, c3 = st.columns(3)
            with c1:
                heart_rate = st.number_input("Heart rate (bpm)", min_value=0.0, step=1.0)
                temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
            with c2:
                blood_pressure = st.text_input("Blood pressure (e.g. 120/80)")
                spo2 = st.number_input("SpO2 (%)", min_value=0.0, max_value=100.0, step=1.0)
            with c3:
                respiratory_rate = st.number_input("Respiratory rate (breaths/min)", min_value=0.0, step=1.0)

            st.markdown("**Physical Exam**")
            general_appearance = st.text_area("General appearance")
            chest_auscultation = st.text_area("Chest auscultation")
            percussion = st.text_area("Percussion")

            st.markdown("**Diagnostics**")
            microbiology = st.text_area("Microbiology")
            abg = st.text_area("Arterial blood gas (ABG)")
            notes = st.text_area("Additional notes")

            submitted_encounter = st.form_submit_button("Save Encounter", type="primary")

        if submitted_encounter:
            payload = {
                "heart_rate": heart_rate or None,
                "blood_pressure": blood_pressure or None,
                "respiratory_rate": respiratory_rate or None,
                "temperature": temperature or None,
                "spo2": spo2 or None,
                "general_appearance": general_appearance or None,
                "chest_auscultation": chest_auscultation or None,
                "percussion": percussion or None,
                "microbiology": microbiology or None,
                "abg": abg or None,
                "notes": notes or None,
            }
            try:
                add_encounter(patient_id, payload)
                st.success("Encounter saved.")
                st.session_state.show_add_encounter = False
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save encounter: {e}")

# --- Appointments -----------------------------------------------------------
with tab_appointments:
    appointments = get_appointments(patient_id)
    if appointments:
        df = pd.DataFrame([{
            "Date": a["appointment_date"][:16].replace("T", " "),
            "Reason": a["reason"],
            "Status": a["status"],
            "Notes": a["notes"],
        } for a in appointments])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No appointments recorded yet.")

    if "show_add_appointment" not in st.session_state:
        st.session_state.show_add_appointment = False
    if st.button("+ Add Appointment"):
        st.session_state.show_add_appointment = not st.session_state.show_add_appointment

    if st.session_state.show_add_appointment:
        with st.form("add_appointment_form"):
            appt_date = st.date_input("Appointment date", value=date.today())
            appt_time = st.time_input("Appointment time")
            reason = st.text_input("Reason")
            status = st.selectbox("Status", ["scheduled", "completed", "cancelled"])
            notes = st.text_area("Notes")
            submitted_appt = st.form_submit_button("Save Appointment", type="primary")

        if submitted_appt:
            appointment_datetime = datetime.combine(appt_date, appt_time)
            payload = {
                "appointment_date": appointment_datetime.isoformat(),
                "reason": reason or None,
                "status": status,
                "notes": notes or None,
            }
            try:
                add_appointment(patient_id, payload)
                st.success("Appointment saved.")
                st.session_state.show_add_appointment = False
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save appointment: {e}")

# --- Treatments ---------------------------------------------------------------
with tab_treatments:
    treatments = get_treatments(patient_id)
    if treatments:
        df = pd.DataFrame([{
            "Start Date": t["start_date"][:10],
            "Description": t["description"],
            "Dosage": t["dosage"],
            "Route": t["route"],
            "Duration": t["duration"],
            "Notes": t["notes"],
        } for t in treatments])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No treatments recorded yet.")

    if "show_add_treatment" not in st.session_state:
        st.session_state.show_add_treatment = False
    if st.button("+ Add Treatment"):
        st.session_state.show_add_treatment = not st.session_state.show_add_treatment

    if st.session_state.show_add_treatment:
        with st.form("add_treatment_form"):
            start_date = st.date_input("Start date", value=date.today())
            description = st.text_input("Treatment / medication *")
            dosage = st.text_input("Dosage")
            route = st.selectbox("Route", ["", "oral", "IV", "IM", "inhaled", "topical", "other"])
            duration = st.text_input("Duration (e.g. 7 days)")
            notes = st.text_area("Notes")
            submitted_treatment = st.form_submit_button("Save Treatment", type="primary")

        if submitted_treatment:
            if not description.strip():
                st.error("Treatment / medication name is required.")
            else:
                payload = {
                    "start_date": datetime.combine(start_date, datetime.min.time()).isoformat(),
                    "description": description,
                    "dosage": dosage or None,
                    "route": route or None,
                    "duration": duration or None,
                    "notes": notes or None,
                }
                try:
                    add_treatment(patient_id, payload)
                    st.success("Treatment saved.")
                    st.session_state.show_add_treatment = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save treatment: {e}")

# --- Predictions (read-only history, created on the Predict page) ----------
with tab_predictions:
    predictions = get_patient_predictions(patient_id)
    if predictions:
        df = pd.DataFrame([{
            "Date": p["created_at"][:16].replace("T", " "),
            "Label": p["label"],
            "Probability": f"{p['probability']:.1%}",
            "WBC": p["wbc"],
            "CRP": p["crp"],
        } for p in predictions])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet -- run one on the Predict page.")