import streamlit as st

from api_client import create_patient

st.title("Create Patient")

with st.form("create_patient_form"):
    name = st.text_input("Full name *")
    col1, col2 = st.columns(2)
    with col1:
        dob = st.date_input("Date of birth", value=None)
    with col2:
        sex = st.selectbox("Sex", ["", "Male", "Female", "Other"])
    contact_info = st.text_input("Contact info")

    st.subheader("Medical History")
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
        # split comma-separated fields into clean lists, dropping empty entries
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
        except Exception as e:
            st.error(f"Failed to create patient: {e}")
