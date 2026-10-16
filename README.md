# Explainable RAG Agentic AI — Clinical Decision Support System

> **Academic Prototype** — Demonstrating how traditional ML disease prediction can be made
> more explainable and clinically informative by combining SHAP, retrieval of clinical
> evidence, and an evidence-grounded LLM workflow.

---

## Project Overview

This system implements an end-to-end pipeline for trustworthy clinical decision support:

```
Patient Input → ML Prediction → SHAP Explainability → Clinical Query
→ ChromaDB RAG → LangChain → LLM → Evidence-Grounded Explanation → Streamlit UI
```

### Disease Models
- **Diabetes**: Age, Gender, BMI, BloodPressure, Insulin, Glucose, DiabetesPedigreeFunction
- **Anemia**: Gender, Hemoglobin, MCH, MCHC, MCV

---

## Project Structure

```
.
├── app.py                        # Streamlit application
├── train_models.py               # Train & save ML models
│
├── models/                       # Saved model files (generated)
│   ├── diabetes_model.pkl
│   ├── diabetes_metadata.json
│   ├── anemia_model.pkl
│   └── anemia_metadata.json
│
├── data/sample_documents/        # Sample knowledge base (replace with real docs)
│   ├── diabetes_sample.txt
│   └── anemia_sample.txt
│
├── rag/                          # RAG pipeline
│   ├── ingest.py                 # Document ingestion (run once)
│   ├── vectorstore.py            # ChromaDB management
│   ├── retriever.py              # Disease-specific retrieval
│   └── prompts.py                # LLM prompt templates
│
├── explainability/
│   └── shap_explainer.py         # SHAP (auto-detects model type)
│
├── prediction/
│   └── predictor.py              # predict_diabetes() / predict_anemia()
│
├── llm/
│   └── generator.py              # LLM integration (configurable provider)
│
├── agent/
│   └── orchestrator.py           # Sequential agentic workflow
│
├── utils/
│   └── helpers.py                # Shared utilities
│
├── chroma_db/                    # ChromaDB persistent store (generated)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example config
cp .env.example .env

# Edit .env and add your API key
# Default provider: OpenAI
OPENAI_API_KEY=sk-your-key-here
LLM_PROVIDER=openai
```

**Supported LLM providers** (edit `LLM_PROVIDER` in `.env`):
| Provider | Environment Variable | Free Tier |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | No |
| `google` | `GOOGLE_API_KEY` | Yes (Gemini) |
| `groq` | `GROQ_API_KEY` | Yes |
| `ollama` | — (local) | Yes (local) |

### 3. Train the ML Models

```bash
python train_models.py
```

This reproduces the exact training logic from the notebooks:
- Loads `diabetes_dataset.csv` and `anemia Dataset.csv`
- Trains Logistic Regression, Random Forest, XGBoost
- Selects best by CV F1-score (ROC-AUC as tiebreaker)
- Saves `models/diabetes_model.pkl` and `models/anemia_model.pkl`

### 4. Ingest Clinical Documents

```bash
python rag/ingest.py
```

This loads the sample clinical documents, splits them into chunks,
generates embeddings (HuggingFace `all-MiniLM-L6-v2`, local/free),
and stores them in ChromaDB.

> **Note:** Embeddings are computed locally — no API key required.

### 5. Launch the Application

```bash
streamlit run app.py
```

---

## Usage

1. Open the Streamlit app in your browser
2. Select a disease (Diabetes or Anemia) in the sidebar
3. Enter patient clinical values
4. Click **Analyze Patient**
5. View results across 4 tabs:
   - **Prediction** — ML model output with probability
   - **Why This Prediction?** — Local & Global SHAP analysis
   - **Clinical Evidence** — Retrieved ChromaDB passages
   - **Explanation** — LLM evidence-grounded explanation

---

## Replacing Sample Documents with Real Documents

The sample documents in `data/sample_documents/` are placeholders for development.
To replace with real clinical documents (e.g., WHO guidelines):

1. Add your `.txt` (or `.pdf` with appropriate loader) files to `data/sample_documents/`
2. Update the `DOCUMENTS` registry in `rag/ingest.py` with the new file paths and metadata
3. Re-run: `python rag/ingest.py --reset`

---

## Architecture

```
STREAMLIT UI
     │
     ▼
Agent Orchestrator (agent/orchestrator.py)
     │
     ├─── Tool 1: Prediction (prediction/predictor.py)
     │           └─ Loads saved .pkl model
     │
     ├─── Tool 2: SHAP (explainability/shap_explainer.py)
     │           └─ Auto-detects model type → correct explainer
     │
     ├─── Tool 3: Query Builder (utils/helpers.py)
     │           └─ Dynamic query from top SHAP features
     │
     ├─── Tool 4: Retrieval (rag/retriever.py)
     │           └─ Disease-specific ChromaDB collection
     │
     └─── Tool 5: LLM (llm/generator.py)
                 └─ Evidence-grounded explanation via strict prompt
```

---

## Important Notes

- **This is an academic prototype.** It is not medically validated.
- The ML prediction should not be treated as a confirmed medical diagnosis.
- SHAP values come from the actual trained model — none are fabricated.
- The LLM is constrained to use only retrieved evidence for clinical claims.
- Sample documents are clearly labelled as development samples — not WHO guidelines.

---

## Disclaimer

> Educational and decision-support prototype. The prediction is generated by a machine-learning
> model and should not be treated as a confirmed medical diagnosis or a substitute for
> professional medical evaluation. Clinical decisions must be made by qualified medical
> professionals with complete patient history and appropriate diagnostic testing.
