"""
core/config.py
--------------
Centralized, strongly typed configuration and logging setup.
Single source of truth for paths, model parameters, and runtime options.
"""

from __future__ import annotations
import os
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "sample_documents"
CHROMA_PATH = BASE_DIR / "chroma_db"


@dataclass(frozen=True)
class DiseaseConfig:
    name: str
    target_column: str
    positive_label: str
    negative_label: str
    feature_names: list[str]
    collection_name: str
    reference_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


DISEASE_CONFIGS: dict[str, DiseaseConfig] = {
    "diabetes": DiseaseConfig(
        name="diabetes",
        target_column="Outcome",
        positive_label="Diabetic",
        negative_label="Non-Diabetic",
        feature_names=[
            "Age",
            "Gender",
            "BMI",
            "BloodPressure",
            "Insulin",
            "Glucose",
            "DiabetesPedigreeFunction",
        ],
        collection_name="diabetes_knowledge",
        reference_ranges={
            "Age": (1.0, 120.0),
            "BMI": (10.0, 70.0),
            "BloodPressure": (30.0, 200.0),
            "Insulin": (0.0, 800.0),
            "Glucose": (40.0, 500.0),
            "DiabetesPedigreeFunction": (0.0, 3.0),
        },
    ),
    "anemia": DiseaseConfig(
        name="anemia",
        target_column="Result",
        positive_label="Anemia",
        negative_label="No Anemia",
        feature_names=[
            "Gender",
            "Hemoglobin",
            "MCH",
            "MCHC",
            "MCV",
        ],
        collection_name="anemia_knowledge",
        reference_ranges={
            "Hemoglobin": (2.0, 25.0),
            "MCH": (10.0, 50.0),
            "MCHC": (15.0, 50.0),
            "MCV": (40.0, 140.0),
        },
    ),
}


@dataclass(frozen=True)
class SystemConfig:
    # Embedding configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    
    # RAG parameters
    default_retrieval_k: int = 4
    min_similarity_threshold: float = 0.35
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    # Model & Explainer cache limits
    explainer_bg_sample_size: int = 200
    
    # Risk Stratification thresholds
    low_risk_threshold: float = 0.35
    high_risk_threshold: float = 0.65


SYSTEM_CONFIG = SystemConfig()


def setup_logger(name: str = "cdss") -> logging.Logger:
    """
    Configure a clean, ASCII-safe logger compatible with Windows cp1252 consoles.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()
