import re

from pypdf import PdfReader


def extract_text(uploaded_file) -> str:
    """Pull raw text out of a .txt or .pdf upload."""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def parse_labs(text: str):
    """Best-effort regex parse of WBC/CRP from free text -- clinician can
    still overwrite these in the editable fields shown alongside it."""
    wbc_match = re.search(r"WBC[:\s]+([\d.]+)", text, re.IGNORECASE)
    crp_match = re.search(r"CRP[:\s]+([\d.]+)", text, re.IGNORECASE)
    wbc = float(wbc_match.group(1)) if wbc_match else None
    crp = float(crp_match.group(1)) if crp_match else None
    return wbc, crp
