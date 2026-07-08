import streamlit as st

from api_client import search_patients, get_patient_predictions, run_agent

st.title("Agent")

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

    # --- 2. Select which past predictions to feed the agent -----------------
    st.header("2. Select Records for the Agent")
    predictions = get_patient_predictions(selected_patient["id"])

    if not predictions:
        st.warning("This patient has no predictions yet -- run one on the Predict page first.")
    else:
        selected_ids = []
        for p in predictions:
            label = (
                f"{p['created_at'][:10]} — {p['label']} ({p['probability']:.1%}) "
                f"— WBC: {p['wbc']}, CRP: {p['crp']}"
            )
            if st.checkbox(label, key=f"pred_{p['id']}"):
                selected_ids.append(p["id"])

        # --- 3. Run agent -----------------------------------------------------
        st.header("3. Generate Report")
        if st.button("Run Agent", type="primary"):
            if not selected_ids:
                st.error("Select at least one record for the agent to reason over.")
            else:
                with st.spinner("Running agent..."):
                    try:
                        report = run_agent(selected_patient["id"], selected_ids)
                        st.subheader("Analysis")
                        st.write(report["analysis"])
                        st.subheader("Reasoning")
                        st.write(report["reasoning"])
                        st.subheader("Report")
                        st.write(report["report_text"])
                    except Exception as e:
                        st.error(f"Agent run failed: {e}")
else:
    st.info("Search for a patient above to begin.")
