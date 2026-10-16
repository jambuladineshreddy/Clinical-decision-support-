"""
core/schemas.py
---------------
Strongly typed domain models, schemas, and contracts.
Supports object attribute access and backward-compatible dictionary indexing.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class ValidationWarning:
    feature: str
    value: Any
    message: str
    severity: str = "warning"  # "warning" | "alert"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult:
    prediction: str
    probability: float
    disease: str
    model_name: str
    risk_tier: str = "Moderate Risk"  # "Low Risk" | "Moderate Risk" | "High Risk"
    warnings: list[ValidationWarning] = field(default_factory=list)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["warnings"] = [w.to_dict() if hasattr(w, "to_dict") else w for w in self.warnings]
        return d


@dataclass
class ShapContribution:
    feature: str
    shap_value: float
    direction: str
    rank: int = 1
    abs_impact: float = 0.0

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShapExplanation:
    feature_contributions: list[dict[str, Any]]
    base_value: float
    model_output: float
    additivity_error: float = 0.0
    is_additivity_verified: bool = True

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceChunk:
    content: str
    source_name: str
    document_name: str
    source_type: str
    disease: str
    chunk_id: int
    score: Optional[float] = None
    similarity_score: float = 0.0
    is_relevance_passed: bool = True

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionProfile:
    prediction_ms: float = 0.0
    shap_ms: float = 0.0
    query_builder_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMExplanationResult:
    explanation: Optional[str]
    shap_summary: str
    evidence_summary: str
    error: Optional[str] = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalAnalysisReport:
    disease: str
    patient_data: dict[str, Any]
    prediction: Optional[dict[str, Any]] = None
    shap_local: Optional[dict[str, Any]] = None
    clinical_query: Optional[str] = None
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    llm_output: Optional[dict[str, Any]] = None
    execution_profile: Optional[dict[str, Any]] = None
    errors: dict[str, str] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
