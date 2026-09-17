# PulmoNetAI

A multimodal clinical decision-support system for pneumonia detection. A fused vision + text + labs + vitals classifier scores a chest X-ray encounter, explains itself with Grad-CAM and modality-attention breakdowns, and hands off to an LLM agent that turns the result into a guideline-grounded respiratory consultation report.

## Project Overview

PulmoNetAI lets a clinician register a patient, upload a chest X-ray alongside free-text notes, WBC/CRP labs, and (optionally) 24 hours of hourly vitals, and get back:

- A **Pneumonia / Normal** prediction with a probability score.
- A **Grad-CAM heatmap** showing where on the X-ray the model focused.
- A breakdown of how much the model relied on each input modality (image vs. text vs. labs vs. vitals) for that specific prediction.
- On request, an AI-generated **Respiratory Consultation Report**: a LangGraph agent retrieves relevant clinical guidelines and peer-reviewed literature (local corpus + live PubMed), cross-checks the patient's values against cited thresholds, and writes a structured, citation-backed report a clinician can review and export to PDF.

The system also tracks a full patient record — encounters, appointments, treatments — behind a FastAPI backend and a Streamlit frontend, and ships a research harness (ablation studies, attention-distribution analysis, latency and emissions benchmarking, report-quality evaluation) used to validate the model and the report pipeline before they're trusted with real cases.

## The Process

1. **Dataset construction** — merge a public chest X-ray dataset with lab metadata and physiologically-plausible vitals sequences into one multimodal, model-ready dataset.
2. **Model training** — freeze pretrained vision/text encoders, cache their embeddings, and train a lightweight fusion head via grid search over learning rate and dropout.
3. **Explainability** — add Grad-CAM for spatial (image) explanations and modality self-attention for "what did the model actually rely on" explanations.
4. **Ablation & attention analysis** — systematically remove modalities to quantify each one's contribution, and analyse the trained model's real attention distribution across the test set.
5. **Agent pipeline** — build a corrective-RAG LangGraph agent that retrieves guideline/literature evidence, grades its own retrieval, rewrites the query if it comes up short, then reasons over and writes a cited clinical report.
6. **Product layer** — wrap all of the above in a FastAPI backend and a Streamlit frontend for patient management, prediction, and report generation.
7. **Evaluation & hardening** — benchmark latency, energy/carbon footprint, and report quality (template adherence, citation rate, diagnostic alignment) to keep the pipeline measurable as it evolves.

## Features

### Dataset creation

`backend/multimodal_dataset_pipeline.py` builds the training set from three sources:

- **Images + clinical metadata**: the HuggingFace dataset [`electricsheepafrica/Multimodal-Chest-X-ray-dataset-for-Normal-and-Bacterial-Pneumonia-in-Africans`](https://huggingface.co), merged against its `Comprehensive_Metadata.csv` (notes, WBC, CRP) by an explicit global index.
- **Vitals**: PhysioNet Sepsis PSV files, forward/back-filled and standardised into 24-hour sequences across 8 channels (`HR, O2Sat, Temp, Resp, SBP, MAP, WBC, FiO2`), pooled and paired onto X-ray samples.
- **Preprocessing**: images through a SwinV2 `AutoImageProcessor`; text (`"Notes: {n}. WBC: {w}. CRP: {c}."`) through a Bio_ClinicalBERT tokenizer.

The result is saved as a HuggingFace `Dataset` to disk, ready for training.

### Training

`backend/model_training.py` freezes the pretrained image (SwinV2-tiny) and text (Bio_ClinicalBERT) encoders, caches their embeddings to disk, then trains only the fusion head over a grid search of learning rates and dropout values (`BCEWithLogitsLoss`, Adam). The best checkpoint by validation loss is saved and evaluated on a held-out test split.

`backend/ablation_study.py` repeats this training process for four modality configurations — Vision Only, Text Only, Vision+Text, and the full Vision+Text+Labs+Vitals model — producing comparative classification reports, confusion matrices, and ROC curves in `backend/ablation_results/`.

`backend/attention_distribution.py` loads the already-trained production checkpoint and, without any retraining, runs a single inference pass over the test set to plot how attention is distributed across the four modalities, both overall and split by diagnosis class (`backend/attention_results/`).

### Running the backend

A FastAPI service (`backend/api/main.py`) exposes:

- `POST/GET/PUT/DELETE /patients` — patient CRUD, plus search, encounters, appointments, and treatments.
- `POST /predict/{patient_id}` — runs the multimodal classifier on an uploaded X-ray + notes + labs (+ optional vitals), generates a Grad-CAM overlay, and persists the prediction.
- `POST /agent/run` — runs the LangGraph agent over one or more of a patient's past predictions and persists the resulting report.

```bash
pip install -r requirements.txt
uvicorn backend.api.main:app --reload
```

### Running the frontend

A Streamlit app (`frontend/`) talks to the backend over HTTP (`frontend/api_client.py`, default `http://localhost:8000`):

- **Add Patient** / **Patients** — register and browse patients.
- **Patient Profile** — view a patient's history, encounters, appointments, and treatments.
- **Predict** — upload an X-ray, notes, labs, and optional vitals CSV, and run a prediction.
- **Agent** — select past predictions and generate a cited, exportable PDF report.

```bash
streamlit run frontend/pulmo_app.py
```

Both services read secrets from a `.env` file (not committed): `OPENAI_API_KEY`, `HF_TOKEN`, `HF_HUB_ENABLE_HF_TRANSFER`, `NCBI_ENTREZ_EMAIL`, `NCBI_API_KEY`.

## Technical Development

**Model architecture** (`backend/models/multimodal_system.py`): frozen SwinV2-tiny and Bio_ClinicalBERT encoders each project into a shared 512-dim space; labs and vitals pass through small 1D-CNNs of their own. The four resulting tokens — image, text, labs, vitals — are fused with multi-head self-attention, mean-pooled, and classified with a single linear layer. When vitals aren't available for a given prediction, the model substitutes a zero token rather than requiring them.

**Explainability**: Grad-CAM (`backend/models/gradcam.py`) hooks the final SwinV2 stage during a gradient-enabled forward/backward pass to produce a real spatial heatmap over the X-ray — the only tensor in the pipeline that still carries per-patch information, since the fusion module pools the image to a single token before attention. The modality self-attention weights are instead used to explain *relative reliance on image vs. text vs. labs vs. vitals*, not spatial location.

**Agent pipeline** (`backend/agent/`): built with **LangGraph** as a corrective-RAG loop — `retrieval → grade_retrieval → (retry via rewrite_query, capped at 3 attempts) → analysis → reasoning → report`. Retrieval draws from a persisted **Chroma** vector store built over local clinical guideline PDFs (BTS CAP guideline, BTS oxygen-use guideline, a pneumonia diagnosis/management guideline) and, optionally, live **PubMed** literature search, selectable per-request as `local`, `pubmed`, or `hybrid`. Retrieved chunks are over-fetched and LLM-reranked before use. The reasoning and report-writing steps run on GPT-4o (auxiliary steps — query condensation, grading, reranking, rewriting — run on the cheaper GPT-4o-mini) and are constrained to cite every guideline-derived claim with a verifiable `[Source: ...]` tag traceable back to a retrieved excerpt; a deterministic post-check catches and corrects citations dropped in the final drafting step.

**Evaluation harness** (`backend/tests/`): mocked wiring tests for fast iteration, opt-in real-API evaluation of all three retrieval modes, a synthetic benchmark set with known ground truth for scoring report quality (template adherence, citation rate, diagnostic alignment), a phase-by-phase latency benchmark across the classifier and agent stages, and CodeCarbon-based energy/emissions profiling (`frontend/tests/power_tests.py`).

## Technologies

- **ML / modelling**: PyTorch, torchvision, Transformers (SwinV2, Bio_ClinicalBERT), scikit-learn, HuggingFace Datasets/Hub
- **Explainability**: custom Grad-CAM (OpenCV for heatmap overlay), modality self-attention analysis
- **Agent / RAG**: LangGraph, LangChain, OpenAI API (GPT-4o, GPT-4o-mini, `text-embedding-3-small`), ChromaDB, PubMed (via `xmltodict`)
- **Backend**: FastAPI, Uvicorn, SQLAlchemy (SQLite), Pydantic
- **Frontend**: Streamlit
- **Reporting**: ReportLab / pypdf (PDF generation and parsing)
- **Evaluation & sustainability**: pytest, spaCy, CodeCarbon (emissions/energy tracking)
- **Data science**: NumPy, Pandas, Matplotlib, Seaborn

## Preview

*Video walkthrough coming soon.*
