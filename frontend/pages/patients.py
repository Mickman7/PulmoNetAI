import pandas as pd
import streamlit as st
from datetime import date


from api_client import list_patients, create_patient

st.title("Patients")

# --- Table of all patients ------------------------------------------------
patients = list_patients()

if not patients:
    st.info("No patients yet -- add one below to get started.")
else:
    rows = []
    for p in patients:
        rows.append({
            "ID": p["id"],
            "Name": p["name"],
            "DOB": p["date_of_birth"],
            "Sex": p["sex"],
            "Smoking Status": p["smoking_status"],
            "Chronic Conditions": ", ".join(p["chronic_conditions"]) if p["chronic_conditions"] else "",
            "Allergies": ", ".join(p["allergies"]) if p["allergies"] else "",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

# --- Add patient (toggleable form) ----------------------------------------
if "show_add_patient_form" not in st.session_state:
    st.session_state.show_add_patient_form = False

if st.button("+ Add Patient"):
    st.session_state.show_add_patient_form = not st.session_state.show_add_patient_form

if st.session_state.show_add_patient_form:
    with st.form("create_patient_form"):
        st.subheader("New Patient")
        name = st.text_input("Full name *")
        col1, col2 = st.columns(2)
        with col1:
            dob = st.date_input("Date of birth", value=None, min_value=date(1900, 1, 1), max_value=date.today(),)
        with col2:
            sex = st.selectbox("Sex", ["", "Male", "Female", "Other"])
        contact_info = st.text_input("Contact info")

        st.markdown("**Medical History**")
        allergies = st.text_input("Allergies (comma-separated)")
        chronic_conditions = st.text_input("Chronic conditions (comma-separated)")
        current_medications = st.text_input("Current medications (comma-separated)")
        past_surgeries = st.text_input("Past surgeries (comma-separated)")
        smoking_status = st.selectbox("Smoking status", ["", "never", "former", "current"])
        family_history = st.text_area("Family history")

        submitted = st.form_submit_button("Create Patient")

    if submitted:
        if not name.strip():
            st.error("Name is required.")
        else:
            def to_list(s):
                return [item.strip() for item in s.split(",") if item.strip()]

            payload = {
                "name": name,
                "date_of_birth": str(dob) if dob else None,
                "sex": sex or None,
                "contact_info": contact_info or None,
                "allergies": to_list(allergies),
                "chronic_conditions": to_list(chronic_conditions),
                "current_medications": to_list(current_medications),
                "past_surgeries": to_list(past_surgeries),
                "smoking_status": smoking_status or None,
                "family_history": family_history or None,
            }

            try:
                patient = create_patient(payload)
                st.success(f"Patient created: {patient['name']} (ID: {patient['id']})")
                st.session_state.show_add_patient_form = False
                st.rerun()  # refresh the table to show the new patient
            except Exception as e:
                st.error(f"Failed to create patient: {e}")