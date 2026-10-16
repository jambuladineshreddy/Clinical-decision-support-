"""
agent/orchestrator.py
---------------------
Sequential agentic pipeline orchestrator with latency profiling and ASCII-safe logging.

Workflow:
    1. Prediction Tool    -> ML model inference + OOD screening
    2. SHAP Tool          -> Local feature attribution
    3. Query Builder Tool -> Clinically contextual retrieval query
    4. Retrieval Tool     -> Disease-specific ChromaDB evidence
    5. LLM Tool           -> Evidence-grounded synthesis
"""

from __future__ import annotations
import time
import traceback
from typing import Any

from core.config import logger, SYSTEM_CONFIG
from core.schemas import ClinicalAnalysisReport, ExecutionProfile
from prediction.predictor import predict_diabetes, predict_anemia
from explainability.shap_explainer import get_local_shap
from rag.retriever import retrieve_evidence
from llm.generator import generate_explanation
from utils.helpers import build_clinical_query


def _tool_predict(disease: str, patient_data: dict[str, Any]) -> dict[str, Any]:
    if disease == "diabetes":
        res = predict_diabetes(patient_data)
    elif disease == "anemia":
        res = predict_anemia(patient_data)
    else:
        raise ValueError(f"Unknown disease identifier: '{disease}'")
    return res.to_dict() if hasattr(res, "to_dict") else res


def _tool_shap(disease: str, patient_data: dict[str, Any]) -> dict[str, Any]:
    return get_local_shap(disease, patient_data)


def _tool_build_query(disease: str, prediction: str, shap_results: dict[str, Any]) -> str:
    top_features = shap_results.get("feature_contributions", [])
    return build_clinical_query(disease, prediction, top_features, n=4)


def _tool_retrieve(disease: str, query: str, k: int = 4) -> list[dict[str, Any]]:
    return retrieve_evidence(disease, query, k=k)


def _tool_llm(
    disease: str,
    prediction_result: dict[str, Any],
    shap_results: dict[str, Any],
    retrieved_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    return generate_explanation(
        disease=disease,
        prediction=prediction_result["prediction"],
        probability=prediction_result["probability"],
        model_name=prediction_result["model_name"],
        shap_results=shap_results,
        retrieved_docs=retrieved_docs,
    )


def run_analysis(
    disease: str,
    patient_data: dict[str, Any],
    retrieval_k: int = 4,
    skip_llm: bool = False,
) -> dict[str, Any]:
    """
    Run complete research-grade clinical decision support pipeline with stage profiling.
    """
    t_start = time.perf_counter()
    profile = ExecutionProfile()
    warnings_list: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    report = ClinicalAnalysisReport(
        disease=disease,
        patient_data=patient_data,
        prediction=None,
        shap_local=None,
        clinical_query=None,
        retrieved_evidence=[],
        llm_output=None,
        execution_profile=None,
        errors={},
        warnings=[],
        success=False,
    )

    # ── Step 1: Prediction & OOD Screening ───────
    t0 = time.perf_counter()
    try:
        pred_dict = _tool_predict(disease, patient_data)
        report.prediction = pred_dict
        if "warnings" in pred_dict and pred_dict["warnings"]:
            warnings_list.extend(pred_dict["warnings"])
        profile.prediction_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"[Step 1: OK] Prediction: {pred_dict['prediction']} ({pred_dict['probability']*100:.1f}%) in {profile.prediction_ms:.1f}ms")
    except Exception as e:
        errors["prediction"] = str(e)
        logger.error(f"[Step 1: FAIL] Prediction failed: {e}")
        report.errors = errors
        report.execution_profile = profile.to_dict()
        return report.to_dict()

    # ── Step 2: Local SHAP Attribution ───────────
    t0 = time.perf_counter()
    try:
        report.shap_local = _tool_shap(disease, patient_data)
        profile.shap_ms = round((time.perf_counter() - t0) * 1000, 2)
        n_feats = len(report.shap_local.get("feature_contributions", []))
        logger.info(f"[Step 2: OK] SHAP attribution ({n_feats} features) in {profile.shap_ms:.1f}ms")
    except Exception as e:
        errors["shap"] = str(e)
        logger.warning(f"[Step 2: FAIL] SHAP attribution failed: {e}")

    # ── Step 3: Clinical Query Builder ───────────
    t0 = time.perf_counter()
    try:
        report.clinical_query = _tool_build_query(
            disease,
            report.prediction["prediction"],
            report.shap_local or {},
        )
        profile.query_builder_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"[Step 3: OK] Clinical query synthesized in {profile.query_builder_ms:.1f}ms")
    except Exception as e:
        errors["query"] = str(e)
        report.clinical_query = f"Clinical significance of key biomarker features in {disease} management."
        logger.warning(f"[Step 3: FALLBACK] Query builder error: {e}")

    # ── Step 4: Vector Evidence Retrieval ────────
    t0 = time.perf_counter()
    try:
        report.retrieved_evidence = _tool_retrieve(
            disease, report.clinical_query, k=retrieval_k
        )
        profile.retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"[Step 4: OK] Retrieved {len(report.retrieved_evidence)} chunks in {profile.retrieval_ms:.1f}ms")
    except Exception as e:
        errors["retrieval"] = str(e)
        logger.warning(f"[Step 4: FAIL] Evidence retrieval error: {e}")

    # ── Step 5: Evidence-Grounded LLM Synthesis ──
    if not skip_llm:
        t0 = time.perf_counter()
        try:
            report.llm_output = _tool_llm(
                disease,
                report.prediction,
                report.shap_local or {"feature_contributions": []},
                report.retrieved_evidence,
            )
            profile.llm_ms = round((time.perf_counter() - t0) * 1000, 2)
            if report.llm_output.get("error"):
                errors["llm"] = report.llm_output["error"]
                logger.warning(f"[Step 5: FAIL] LLM response returned error: {report.llm_output['error']}")
            else:
                logger.info(f"[Step 5: OK] LLM explanation generated in {profile.llm_ms:.1f}ms")
        except Exception as e:
            errors["llm"] = str(e)
            report.llm_output = {"explanation": None, "error": str(e)}
            logger.warning(f"[Step 5: FAIL] LLM synthesis failed: {e}")

    profile.total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    report.execution_profile = profile.to_dict()
    report.errors = errors
    report.warnings = warnings_list
    report.success = report.prediction is not None

    return report.to_dict()
