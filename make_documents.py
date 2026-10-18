from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
EVIDENCE = ROOT / "evidence"
DOCS.mkdir(exist_ok=True)
EVIDENCE.mkdir(exist_ok=True)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"],
    alignment=TA_CENTER, fontSize=19, leading=24, spaceAfter=16
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading2"],
    fontSize=13, leading=16, spaceBefore=12, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="SmallBody", parent=styles["BodyText"],
    fontSize=9, leading=13, spaceAfter=6
))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2 * cm, 1.2 * cm, "AI-A2-7196 | Musanze Harvest Intelligence")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()

def make_pdf(path, story):
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=1.8*cm, bottomMargin=2*cm
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

def P(text, style="SmallBody"):
    return Paragraph(text, styles[style])

def section(title, body):
    return [P(title, "Section"), P(body)]

# -------------------- Assignment report --------------------
story = [
    Spacer(1, 2*cm),
    P("MUSANZE HARVEST INTELLIGENCE", "ReportTitle"),
    P("SWE3513 — AI Assignment 2", "Heading2"),
    Spacer(1, 0.4*cm),
    P("<b>Student:</b> Cedrick Manzi"),
    P("<b>Verification code:</b> AI-A2-7196"),
    P("<b>Dataset:</b> data/AI_A2_2527196.csv"),
    P("<b>Report date:</b> 17 October 2026"),
    Spacer(1, 0.6*cm),
    P("Project Summary", "Section"),
    P(
        "Musanze Harvest Intelligence is a Python machine-learning project "
        "that validates agricultural data, trains regression and classification "
        "models, performs clustering, saves trained models, and exposes "
        "predictions through a FastAPI application."
    ),
    P("1. Objectives", "Section"),
    P(
        "The project aims to prepare a farm dataset for analysis, compare "
        "candidate machine-learning models, save the selected models, and "
        "provide a programmatic interface for predictions. The implementation "
        "also includes automated API tests."
    ),
    P("2. Dataset and preparation", "Section"),
    P(
        "The generated dataset contains 264 rows before duplicate removal. "
        "Validation identified 3 duplicate rows; the cleaned dataset contained "
        "261 rows. Missing values were reported in soil_moisture_pct (13) and "
        "fertilizer_kg (13). The preprocessing workflow uses median imputation, "
        "scaling, and one-hot encoding for the season feature."
    ),
    P("3. Regression results", "Section"),
    P(
        "Linear Regression was selected from the evaluated regression models. "
        "On the reported hold-out evaluation, it achieved MAE = 0.88167, "
        "RMSE = 1.14908, and R-squared = 0.80443. The evaluated Random Forest "
        "regressor achieved MAE = 1.10749, RMSE = 1.33847, and "
        "R-squared = 0.73465."
    ),
    P("4. Classification results", "Section"),
    P(
        "Logistic Regression was selected for classification. Reported metrics "
        "were accuracy = 0.96226, precision = 0.96154, recall = 0.96154, "
        "F1-score = 0.96154, and ROC AUC = 0.99430. The Random Forest "
        "classifier reported an F1-score of 0.92308. These values reflect "
        "the project's evaluation run and should not be interpreted as a "
        "guarantee of performance on new data."
    ),
    P("5. Clustering results", "Section"),
    P(
        "The clustering workflow compared candidate cluster counts from k=2 "
        "through k=6 using silhouette score. The best reported result was "
        "k=2 with a silhouette score of approximately 0.15057. This indicates "
        "some cluster structure, but the score is modest and the groups should "
        "be interpreted cautiously."
    ),
    P("6. API and saved models", "Section"),
    P(
        "The FastAPI application provides the root route, /health, /model-info, "
        "and /predict. The prediction input includes rainfall, average "
        "temperature, soil moisture, fertilizer quantity, field size, season, "
        "and cooperative support. Input validation checks constraints such as "
        "non-negative rainfall, soil moisture between 0 and 100, positive field "
        "size, supported season values A or B, and a binary cooperative-support "
        "value. Trained models and the clustering preprocessor are saved under "
        "the models directory."
    ),
    P("7. Testing", "Section"),
    P(
        "The project's API tests cover the health endpoint, model information, "
        "a valid prediction request, and rejection of an invalid season. The "
        "latest recorded command was python -m pytest -q, which reported "
        "4 passed and 1 warning. The warning concerned Starlette's test-client "
        "integration with httpx; it did not cause the tests to fail."
    ),
    P("8. Limitations and future improvements", "Section"),
    P(
        "The models are trained on a generated dataset, so their results may "
        "not represent real farm conditions. Future work should evaluate the "
        "models on independently collected agricultural data, check for data "
        "leakage, use cross-validation, improve clustering interpretation, "
        "and monitor model performance after deployment."
    ),
    P("9. Conclusion", "Section"),
    P(
        "The project implements an end-to-end machine-learning workflow with "
        "data validation, preprocessing, model evaluation, model persistence, "
        "a prediction API, and automated tests. Results should be assessed in "
        "the context of the generated dataset and the evaluation procedure."
    ),
]
make_pdf(DOCS / "AI_A2_2527196_REPORT.pdf", story)

# -------------------- Test log --------------------
test_story = [
    P("AUTOMATED TEST LOG", "ReportTitle"),
    P("<b>Project:</b> Musanze Harvest Intelligence"),
    P("<b>Verification code:</b> AI-A2-7196"),
    P("<b>Command:</b> python -m pytest -q"),
    Spacer(1, 0.3*cm),
    P("Recorded result", "Section"),
]
data = [
    ["Item", "Recorded result"],
    ["Health endpoint test", "Passed"],
    ["Model information endpoint test", "Passed"],
    ["Valid prediction request test", "Passed"],
    ["Invalid season validation test", "Passed"],
    ["Total", "4 passed, 1 warning"],
]
table = Table(data, colWidths=[8*cm, 7*cm], repeatRows=1)
table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#24476B")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
    ("PADDING", (0,0), (-1,-1), 7),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
]))
test_story += [
    table, Spacer(1, 0.4*cm),
    P(
        "Warning noted during the recorded run: Starlette emitted a "
        "deprecation warning regarding its use with httpx. The warning did "
        "not prevent the tests from passing."
    ),
    P(
        "This document records the result supplied for the latest test run. "
        "Rerun the command before submission if the code changes."
    ),
]
make_pdf(EVIDENCE / "AI_A2_2527196_TEST_LOG.pdf", test_story)

# -------------------- Declaration template --------------------
declaration_story = [
    P("STUDENT DECLARATION", "ReportTitle"),
    P("<b>Project:</b> Musanze Harvest Intelligence"),
    P("<b>Course:</b> SWE3513 — AI Assignment 2"),
    P("<b>Verification code:</b> AI-A2-7196"),
    Spacer(1, 0.4*cm),
    P(
        "I, Cedrick Manzi, declare that I have reviewed the submitted "
        "assignment and that the work is presented honestly. Any external "
        "sources, assistance, or tools used should be acknowledged according "
        "to the course requirements. I understand that I am responsible for "
        "the accuracy of this declaration and the submitted work."
    ),
    Spacer(1, 0.8*cm),
    P("Student signature: __________________________________"),
    Spacer(1, 0.4*cm),
    P("Date: ______________________________________________"),
    Spacer(1, 0.8*cm),
    P(
        "<b>Important:</b> Review and edit this declaration to match your "
        "institution's required wording and your actual work before signing "
        "or submitting it."
    ),
]
make_pdf(EVIDENCE / "AI_A2_2527196_DECLARATION.pdf", declaration_story)

# -------------------- AI use statement --------------------
ai_use = """# AI Use Statement

**Project:** Musanze Harvest Intelligence  
**Course:** SWE3513 — AI Assignment 2  
**Verification code:** AI-A2-7196

## Assistance and verification

AI assistance was used during project development for guidance on Python,
machine-learning workflow organization, API implementation, testing,
debugging, and preparation of documentation.

The student should review the final source code, generated results, and
documentation, and should disclose tools and assistance in the format required
by the course or institution.

## Student confirmation

Before submission, confirm that this statement accurately describes the
assistance used. Add any required tool names, dates, prompts, or further
details required by the assignment instructions. Do not claim that code or
results were independently produced unless that is accurate.
"""
(EVIDENCE / "AI_USE.md").write_text(ai_use, encoding="utf-8")

print("Created:")
for path in [
    DOCS / "AI_A2_2527196_REPORT.pdf",
    EVIDENCE / "AI_A2_2527196_TEST_LOG.pdf",
    EVIDENCE / "AI_A2_2527196_DECLARATION.pdf",
    EVIDENCE / "AI_USE.md",
]:
    print(f"{path.relative_to(ROOT)}: {path.exists()}")
