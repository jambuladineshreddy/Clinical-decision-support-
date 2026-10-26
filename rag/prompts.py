"""
rag/prompts.py
--------------
Strict LLM prompt template for evidence-grounded clinical explanation.

The LLM MUST:
  - Summarise the model prediction
  - Explain the SHAP feature contributions in plain language
  - Use ONLY the retrieved clinical evidence for clinical claims
  - Clearly label sections
  - Include a safety disclaimer

The LLM MUST NOT:
  - Diagnose independently
  - Override the ML model
  - Invent medical evidence or citations
  - Prescribe medication
  - Claim a feature caused the disease
  - Present the ML prediction as a confirmed diagnosis
"""

from langchain_core.prompts import PromptTemplate


CLINICAL_EXPLANATION_TEMPLATE = """
You are a clinical decision support assistant. Your role is to explain a machine learning model prediction using SHAP feature contributions and retrieved clinical evidence.

STRICT INSTRUCTIONS:
- Use only the retrieved clinical evidence provided below for any clinical claims.
- Do NOT invent medical information, citations, or guidelines.
- Do NOT diagnose the patient independently.
- Do NOT prescribe medication or specific treatments.
- Do NOT claim that any feature caused the disease. Instead say "contributed to the model's prediction."
- Do NOT present the ML prediction as a confirmed medical diagnosis.
- If the retrieved evidence does not contain enough information to address a clinical question, explicitly state: "The available retrieved evidence does not provide sufficient information on this point."
- Use language such as: "The model predicted...", "According to the retrieved evidence...", "The features that contributed most to this prediction were..."

---

PATIENT ANALYSIS REPORT
========================

DISEASE: {disease}
ML MODEL PREDICTION: {prediction}
PREDICTION PROBABILITY: {probability}
MODEL USED: {model_name}

---

TOP CONTRIBUTING FEATURES (from SHAP analysis):
{shap_summary}

---

RETRIEVED CLINICAL EVIDENCE:
{retrieved_evidence}

---

Please generate a structured clinical decision support report with the following sections. Use exactly these section headers:

## A. PREDICTION SUMMARY
Briefly summarise what the ML model predicted and the confidence level.
Remind the reader that this is a model prediction, not a confirmed diagnosis.

## B. MODEL EXPLANATION (SHAP)
Explain what the top SHAP features mean in plain language.
For each top feature, describe how it contributed to the model's output (positive = pushed toward {positive_class}, negative = pushed away from {positive_class}).
State clearly: these are model contributions, not causal medical findings.

## C. CLINICAL CONTEXT
Using ONLY the retrieved clinical evidence above, provide brief clinical context for the top contributing features.
If a feature is not covered in the retrieved evidence, say so explicitly.
Cite the evidence as: [Retrieved from: Development Sample Clinical Document]

## D. EVIDENCE-BASED DECISION SUPPORT
Based solely on the retrieved clinical evidence, provide general educational information that may be relevant to clinical decision-making.
Do NOT recommend specific treatments or medications.

## E. SAFETY NOTE
Include a mandatory safety note stating:
- This is an academic prototype for educational purposes only.
- The ML prediction should not be treated as a confirmed medical diagnosis.
- Clinical decisions must be made by qualified medical professionals with complete patient history and appropriate laboratory testing.
"""

CLINICAL_PROMPT = PromptTemplate(
    input_variables=[
        "disease",
        "prediction",
        "probability",
        "model_name",
        "shap_summary",
        "retrieved_evidence",
        "positive_class",
    ],
    template=CLINICAL_EXPLANATION_TEMPLATE,
)


def format_shap_summary(feature_contributions: list[dict], n: int = 5) -> str:
    """Format top-n SHAP features into a text summary for the prompt."""
    top = feature_contributions[:n]
    lines = []
    for c in top:
        val = c.get("shap_value", 0.0)
        direction = c.get("direction")
        if not direction:
            direction = "increases risk" if val > 0 else "decreases risk"
        lines.append(
            f"  - {c.get('feature', 'Unknown')}: SHAP value = {val:+.4f} ({direction})"
        )
    return "\n".join(lines)


def format_retrieved_evidence(evidence_list: list[dict]) -> str:
    """Format retrieved evidence chunks into a text block for the prompt."""
    if not evidence_list:
        return "No clinical evidence was retrieved for this query."

    parts = []
    for i, ev in enumerate(evidence_list, start=1):
        content = ev.get("content") or ev.get("text", "")
        source = ev.get("source_name") or ev.get("source", "Clinical Guidelines")
        doc = ev.get("document_name") or ev.get("disease", "Reference Document")
        parts.append(
            f"[Evidence {i}]\n"
            f"Source: {source}\n"
            f"Document: {doc}\n"
            f"Passage:\n{content.strip()}\n"
        )
    return "\n---\n".join(parts)


POSITIVE_CLASS_MAP = {
    "diabetes": "Diabetic",
    "anemia"  : "Anemia",
}
