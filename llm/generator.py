"""
llm/generator.py
----------------
Evidence-grounded clinical explanation synthesis using configurable LLM providers.
Enforces strict hallucination guardrails and prompt safety constraints.

Providers supported:
    - openai (default)
    - google
    - groq
    - ollama
"""

from __future__ import annotations
import os
import threading
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

from core.config import logger
from core.schemas import LLMExplanationResult
from rag.prompts import (
    CLINICAL_PROMPT,
    format_shap_summary,
    format_retrieved_evidence,
    POSITIVE_CLASS_MAP,
)

_llm_lock = threading.Lock()
_llm_singleton: Any = None


def _build_llm():
    """Build or retrieve cached LLM instance."""
    global _llm_singleton
    if _llm_singleton is not None:
        return _llm_singleton

    with _llm_lock:
        if _llm_singleton is not None:
            return _llm_singleton

        provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY not configured. Set it in .env or choose another provider."
                )
            _llm_singleton = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                temperature=0.1,  # Low temperature for clinical consistency
                openai_api_key=api_key,
                timeout=30.0,
            )

        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY not set.")
            _llm_singleton = ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_MODEL", "gemini-3.8-flash"),
                temperature=0.1,
                google_api_key=api_key,
                transport="rest",  # Use plain HTTPS instead of gRPC (gRPC blocked on hotspots/firewalls)
            )

        elif provider == "groq":
            from langchain_groq import ChatGroq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY not set.")
            _llm_singleton = ChatGroq(
                model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
                temperature=0.1,
                groq_api_key=api_key,
            )

        elif provider == "ollama":
            from langchain_community.llms import Ollama
            _llm_singleton = Ollama(
                model=os.getenv("OLLAMA_MODEL", "llama3"),
                temperature=0.1,
            )

        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: '{provider}'. "
                "Supported: openai, google, groq, ollama."
            )

        logger.info(f"Initialized cached LLM client for provider: {provider}")
        return _llm_singleton


def generate_explanation(
    disease: str,
    prediction: str,
    probability: float,
    model_name: str,
    shap_results: dict[str, Any],
    retrieved_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate evidence-grounded clinical explanation.
    """
    try:
        llm = _build_llm()

        shap_summary = format_shap_summary(
            shap_results.get("feature_contributions", [])
        )
        evidence_summary = format_retrieved_evidence(retrieved_docs)
        positive_class = POSITIVE_CLASS_MAP.get(disease, "Positive")
        prob_pct = f"{probability * 100:.1f}%"

        prompt_text = CLINICAL_PROMPT.format(
            disease=disease.capitalize(),
            prediction=prediction,
            probability=prob_pct,
            model_name=model_name,
            shap_summary=shap_summary,
            retrieved_evidence=evidence_summary,
            positive_class=positive_class,
        )

        response = llm.invoke(prompt_text)
        explanation_text = response.content if hasattr(response, "content") else str(response)

        result = LLMExplanationResult(
            explanation=explanation_text,
            shap_summary=shap_summary,
            evidence_summary=evidence_summary,
            error=None,
        )
        return result.to_dict()

    except Exception as e:
        logger.warning(f"LLM explanation generation aborted: {e}")
        result = LLMExplanationResult(
            explanation=None,
            shap_summary="",
            evidence_summary="",
            error=str(e),
        )
        return result.to_dict()
