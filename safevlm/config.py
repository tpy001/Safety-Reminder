"""Portable defaults; local paths are resolved on the caller's machine."""
import os
from pathlib import Path

MODEL_PATH = os.environ.get("SAFEVLM_MODEL_PATH", "models/llava-1.5-7b")
RESOURCES = Path(__file__).resolve().parent / "resources"
DATASET_REPO = "PeiYuan001/VLM-Safety-Datasets"
DATASET_REVISION = "48cf6aa39472c962ddd0b2d627cbf23ea5023b48"
