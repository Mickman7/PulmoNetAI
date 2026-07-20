"""
Turns the agent's plain-text report into a formatted PDF, in memory (no temp
files needed) so Streamlit can embed + offer it for download in one pass.
"""

import io
import re

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors

# matches numbered section headers like "1. PATIENT DEMOGRAPHICS & VITALS"
_SECTION_RE = re.compile(r"^\d+\.\s+[A-Z][A-Z\s&/]+$")
# matches "## Heading" style (older SOAP template, kept for compatibility)
_MD_HEADING_RE = re.compile(r"^#{1,3}\s+")
# matches "Label: value" lines so the label can be bolded
_FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z\s()/-]{1,40}):\s*(.*)$")


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=16, leading=20, spaceAfter=4,
        textColor=colors.HexColor("#0f766e"), fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", fontSize=12, leading=16, spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#0f766e"), fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="FieldLabel", fontSize=10, leading=14, spaceAfter=2,
        fontName="Helvetica",
    ))
    return styles


def generate_pdf_bytes(report_text: str, patient_name: str = "") -> bytes:
    """Renders report_text (plain text with numbered/markdown section headers
    and 'Label: value' field lines) into a PDF, returned as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=20 * mm, bottomMargin=20 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
    )
    styles = _build_styles()
    story = []

    story.append(Paragraph("Respiratory Consultation Report", styles["ReportTitle"]))
    if patient_name:
        story.append(Paragraph(f"Patient: {patient_name}", styles["Normal"]))
    story.append(Spacer(1, 10))

    for raw_line in report_text.splitlines():
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 6))
            continue

        if _SECTION_RE.match(line) or _MD_HEADING_RE.match(line):
            clean = _MD_HEADING_RE.sub("", line)
            story.append(Paragraph(clean, styles["SectionHeading"]))
            continue

        field_match = _FIELD_RE.match(line)
        if field_match:
            label, value = field_match.groups()
            value = value if value else "Not available/Not assessed"
            story.append(Paragraph(f"<b>{label}:</b> {value}", styles["FieldLabel"]))
            continue

        # bullet-style lines (e.g. "- Heart Rate: ...")
        if line.startswith("- "):
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{line[2:]}", styles["FieldLabel"]))
            continue

        story.append(Paragraph(line, styles["Normal"]))

    doc.build(story)
    return buffer.getvalue()