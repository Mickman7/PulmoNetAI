import pandas as pd
import streamlit as st

from api_client import list_patients

st.title("Patients")

if st.button("+ Add Patient", type="primary"):
    st.switch_page("pages/add_patient.py")  # <-- rename to match your actual filename

patients = list_patients()

if not patients:
    st.info("No patients yet -- click '+ Add Patient' to get started.")
else:
    rows = []
    for p in patients:
        rows.append({
            "ID": p["id"],
            "Name": p["name"],
            "DOB": p["date_of_birth"],
            "Sex": p["sex"],
            "Contact": p["contact_info"],
            "Smoking Status": p["smoking_status"],
            "Chronic Conditions": ", ".join(p["chronic_conditions"]) if p["chronic_conditions"] else "",
            "Allergies": ", ".join(p["allergies"]) if p["allergies"] else "",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("View Patient Profile")
    options = {f"{p['name']} (ID: {p['id']})": p["id"] for p in patients}
    choice = st.selectbox("Select a patient", list(options.keys()))
    if st.button("View Profile"):
        st.session_state["selected_patient_id"] = options[choice]
        st.switch_page("pages/patient_profile.py")  # <-- rename to match your actual filename