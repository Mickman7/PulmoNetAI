import pandas as pd
import streamlit as st

from api_client import list_patients, delete_patient  # Ensure delete_patient is imported from api_client

st.title("Patients")

if st.button("+ Add Patient", type="primary"):
    st.switch_page("pages/add_patient.py")

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
    st.subheader("Manage Patient")
    
    options = {f"{p['name']} (ID: {p['id']})": p["id"] for p in patients}
    choice = st.selectbox("Select a patient", list(options.keys()))
    selected_id = options[choice]

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("View Profile", use_container_width=True):
            st.session_state["selected_patient_id"] = selected_id
            st.switch_page("pages/patient_profile.py")

    with col2:
        if st.button("Delete Patient", type="secondary", use_container_width=True):
            st.session_state["confirm_delete_id"] = selected_id

    # Confirmation step to prevent accidental deletion
    if st.session_state.get("confirm_delete_id") == selected_id:
        st.warning(f"Are you sure you want to delete patient ID {selected_id}? This action cannot be undone.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, Confirm Delete", type="primary"):
                try:
                    delete_patient(selected_id)
                    st.success("Patient deleted successfully.")
                    del st.session_state["confirm_delete_id"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete patient: {e}")
        with c2:
            if st.button("Cancel"):
                del st.session_state["confirm_delete_id"]
                st.rerun()