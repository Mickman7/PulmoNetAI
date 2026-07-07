"""
Pulmo — Multimodal Pneumonia Detection App (Streamlit)

Run with: streamlit run pulmo_app.py

Before running:
- Set MODEL_PATH below to your saved .pth file.
- Make sure torch, transformers, pypdf, streamlit are installed in this environment.
"""

import re

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

MODEL_PATH = "/Users/mickman/Documents/programs/PulmoNetAI/backend/models/multimodal_pneumonia_model.pth"  
IMAGE_MODEL_NAME = "microsoft/swinv2-tiny-patch4-window8-256"
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


# ---------------------------------------------------------------------------
# Model definition (must match training exactly, or load_state_dict will fail)
# ---------------------------------------------------------------------------
class MultimodalSystem(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.3, freeze_encoders=True):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(IMAGE_MODEL_NAME)
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME)

        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.img_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)  # outputs raw logits

    def forward(self, images, text_input, return_attention=False):
        img_features = self.image_encoder(images).last_hidden_state  # [B, num_patches, hidden]
        text_features = self.text_encoder(**text_input).pooler_output  # [B, hidden]

        img_vector = self.img_proj(img_features)
        text_vector = self.text_proj(text_features)

        query = text_vector.unsqueeze(1)
        attention_out, attn_weights = self.fusion(query, img_vector, img_vector)
        fused_vec = self.norm(attention_out.squeeze(1) + text_vector)
        fused_vec = self.dropout(fused_vec)
        logits = self.classifier(fused_vec)

        if return_attention:
            return logits, attn_weights
        return logits


# ---------------------------------------------------------------------------
# Cached loaders — @st.cache_resource keeps these in memory across reruns,
# so the model/tokenizer aren't reloaded every time a widget changes.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = MultimodalSystem(freeze_encoders=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_processors():
    img_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    return img_processor, tokenizer


# ---------------------------------------------------------------------------
# File parsing helpers
# ---------------------------------------------------------------------------
def extract_text(uploaded_file):
    """Pull raw text out of a .txt or .pdf upload."""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def parse_labs(text):
    """Best-effort regex parse of WBC/CRP from free text.
    Clinicians can still overwrite these in the editable fields below."""
    wbc_match = re.search(r"WBC[:\s]+([\d.]+)", text, re.IGNORECASE)
    crp_match = re.search(r"CRP[:\s]+([\d.]+)", text, re.IGNORECASE)
    wbc = float(wbc_match.group(1)) if wbc_match else None
    crp = float(crp_match.group(1)) if crp_match else None
    return wbc, crp


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Pulmo", page_icon="🫁")
st.title("🫁 Pulmo — Pneumonia Detection Assistant")
st.caption("Multimodal chest X-ray + clinical note screening tool")

st.header("1. Chest X-ray")
image_file = st.file_uploader("Upload a radiograph (PNG/JPEG)", type=["png", "jpg", "jpeg"])
if image_file:
    st.image(image_file, caption="Uploaded radiograph", width=300)

st.header("2. Clinical Notes & Lab Results")
manual_entry = st.checkbox("Enter manually instead of uploading a file")

notes, wbc, crp = "", None, None

if manual_entry:
    notes = st.text_area("Clinical notes", placeholder="e.g. Productive cough, fever, dyspnoea...")
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
            st.info("Couldn't auto-detect WBC/CRP from the file — please check the values above.")

st.header("3. Prediction")

if st.button("Predict", type="primary"):
    if image_file is None:
        st.error("Please upload a radiograph image.")
    elif not notes.strip():
        st.error("Please provide clinical notes (upload or manual entry).")
    else:
        with st.spinner("Loading model and running prediction..."):
            model = load_model()
            img_processor, tokenizer = load_processors()

            img = Image.open(image_file).convert("RGB")
            pixel_values = img_processor(img, return_tensors="pt")["pixel_values"]

            text_str = f"Notes: {notes}. WBC: {wbc}. CRP: {crp}."
            text_inputs = tokenizer(text_str, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                logits = model(pixel_values, text_inputs)
                probability = torch.sigmoid(logits).item()

        prediction = "Pneumonia" if probability > 0.5 else "Normal"

        st.subheader("Result")
        col1, col2 = st.columns(2)
        col1.metric("Prediction", prediction)
        col2.metric("Probability", f"{probability:.1%}")
        st.progress(probability)

        st.caption(
            "This is a decision-support tool, not a diagnostic replacement. "
            "All results should be reviewed by a qualified clinician."
        )