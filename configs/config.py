"""
Global configuration for hallucination detection.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ==================== GPT Judge ====================
GPT_JUDGE_MODEL = "gpt-5.5"
GPT_JUDGE_TEMPERATURE = 0.0

# ==================== Model API Configs ====================
MODEL_API_CONFIGS = {
    "gpt-5.4-mini": {
        "api_key": os.environ.get("GPT_API_KEY", ""),
        "base_url": "https://bscclheilkfs.sealosbja.site/v1",
        "api_method": "responses",
    },
    "gpt-5.5": {
        "api_key": os.environ.get("GPT_API_KEY", ""),
        "base_url": "https://bscclheilkfs.sealosbja.site/v1",
        "api_method": "responses",
    },
    "Qwen3.5-35B-A3B": {
        "api_key": os.environ.get("QWEN_API_KEY", ""),
        "base_url": "https://yunwu.ai/v1",
        "api_method": "chat",
        "api_model": "qwen3.5-35b-a3b",
    },
    "Qwen3-VL-235B-A22B-Instruct": {
        "api_key": os.environ.get("QWEN_VL_API_KEY", ""),
        "base_url": "https://www.sophnet.com/api/open-apis/v1",
        "api_method": "chat",
        "api_model": "Qwen3-VL-235B-A22B-Instruct",
    },
    "gemini-2.5-flash": {
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "base_url": "https://yunwu.ai/v1",
        "api_method": "chat",
        "api_model": "gemini-2.5-flash-all",
    },
}


def resolve_model_api_config(model_name: str) -> dict:
    model_config = MODEL_API_CONFIGS.get(model_name)
    if model_config is None:
        known = ", ".join(sorted(MODEL_API_CONFIGS))
        raise ValueError(
            f"No API config found for model {model_name!r}. "
            f"Add it to config.MODEL_API_CONFIGS. Known models: {known}"
        )

    resolved_api_method = model_config["api_method"]
    if resolved_api_method not in {"chat", "responses", "messages"}:
        raise ValueError("api_method must be 'chat', 'responses', or 'messages'")

    resolved_api_key = model_config.get("api_key", "")
    if not resolved_api_key or resolved_api_key == "sk-your-key-here":
        raise ValueError(
            f"No API key configured for model {model_name!r}. "
            "Update config.MODEL_API_CONFIGS."
        )

    resolved_base_url = model_config.get("base_url", "")
    if not resolved_base_url:
        raise ValueError(
            f"No base URL configured for model {model_name!r}. "
            "Update config.MODEL_API_CONFIGS."
        )

    return {
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "api_method": resolved_api_method,
        "api_model": model_config.get("api_model", model_name),
        "max_tokens": model_config.get("max_tokens"),
        "thinking_budget": model_config.get("thinking_budget"),
    }

# ==================== Data Paths ====================
# Each directory points to the HF Datasets parquet location
POPE_DATA_DIR = str(_PROJECT_ROOT / "data" / "POPE" / "data")
MATHVISTA_DATA_DIR = str(_PROJECT_ROOT / "data" / "MathVista" / "data")
OCR_DATA_DIR = str(_PROJECT_ROOT / "data" / "OCRBench" / "data")

POPE_DEFAULT_MAX_SAMPLES = 1000
POPE_SAMPLE_SEED = 42

# ==================== Models to Evaluate ====================
MODELS = [
    "gpt-5.4-mini",
    "gemini-2.5-flash",
    "Qwen3.5-35B-A3B",
    "Qwen3-VL-235B-A22B-Instruct",
]

# ==================== Output ====================
OUTPUT_DIR = "results"