"""
Reusable resumable batch execution helpers.
"""

import base64
import json
import logging
import mimetypes
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, TypeVar

from PIL import Image


SAVE_EVERY = 1
LOG_EVERY = 10
LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def hf_image_to_tempfile(image_data, suffix=".jpg") -> str:
    """
    Extract image bytes from a Hugging Face Datasets parquet column
    and write to a persistent temp file. Returns the file path.

    Handles:
    - dict with 'bytes' key (HF Image feature in parquet)
    - raw bytes (binary column)
    - string path (pass-through, already on disk)
    """
    if image_data is None:
        return ""
    if isinstance(image_data, dict):
        image_bytes = image_data.get("bytes", b"")
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    elif isinstance(image_data, str):
        return image_data  # already a file path
    else:
        return ""
    if not image_bytes:
        return ""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(image_bytes)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------

def sort_json_keys(keys):
    """Sort numeric string IDs numerically and other IDs lexicographically."""
    return sorted(
        keys,
        key=lambda item: (0, int(item)) if item.isdigit() else (1, item),
    )


def load_json_object(path: str) -> dict:
    """Load a JSON object, returning an empty dict when the file is absent."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def save_json_atomic(data: dict, path: str) -> None:
    """Atomically write a JSON object."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)


def load_text_map(path: str) -> dict[str, str]:
    """Load a JSON object as a string-to-string mapping."""
    data = load_json_object(path)
    return {str(k): "" if v is None else str(v) for k, v in data.items()}


def load_existing_details(filepath: str, id_field: str, allowed_ids: set[str]) -> dict[str, dict]:
    """Load previously saved per-sample details for resumable judging."""
    data = load_json_object(filepath)
    details = data.get("details", [])
    if not isinstance(details, list):
        return {}
    return {
        str(d[id_field]): d
        for d in details
        if isinstance(d, dict) and id_field in d and str(d[id_field]) in allowed_ids
    }


def load_response_subset(
    path: str,
    samples: list[dict],
    id_key: str,
    dataset: str,
) -> tuple[dict[str, str], list[dict]]:
    """Load responses and keep only samples with available model outputs."""
    if not path:
        raise ValueError(f"No response file provided for {dataset}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Response file not found: {path}")

    responses = load_text_map(path)
    filtered = [
        sample
        for sample in samples
        if str(sample.get(id_key, "")) in responses
    ]
    if not filtered:
        raise ValueError(
            f"Response file {path} has no matching {dataset} responses "
            f"for the loaded samples."
        )

    missing = len(samples) - len(filtered)
    LOGGER.info(
        "  Loaded %s/%s matching %s responses from %s",
        len(filtered), len(samples), dataset, path,
    )
    if missing:
        LOGGER.info("  Skipping %s samples without responses", missing)
    return responses, filtered


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def guess_image_media_type(image_path: str) -> str:
    """Detect actual image format from file content using PIL."""
    try:
        with Image.open(image_path) as im:
            fmt = (im.format or "jpeg").lower()
        return f"image/{fmt}"
    except Exception:
        mime, _ = mimetypes.guess_type(image_path)
        return mime or "image/jpeg"


def encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_data_url(image_path: str) -> str:
    """Build a data URL with the actual image format."""
    mime_type = guess_image_media_type(image_path)
    return f"data:{mime_type};base64,{encode_image_base64(image_path)}"


# ---------------------------------------------------------------------------
# Resumable batch processing
# ---------------------------------------------------------------------------

def run_resumable_batch(
    *,
    items: list[dict],
    item_id: Callable[[dict], str],
    completed: dict[str, T],
    process_one: Callable[[dict], tuple[str, T]],
    save_completed: Callable[[dict[str, T]], None],
    workers: int,
    label: str,
) -> dict[str, T]:
    """Run items with fixed resume/save/log strategy and optional parallelism."""
    if workers <= 0:
        raise ValueError("--workers must be positive")

    pending = [
        item for item in items
        if str(item_id(item)) not in completed
    ]
    skipped = len(items) - len(pending)
    if completed:
        LOGGER.info("Resume: loaded %s existing %s", len(completed), label)
    if skipped:
        LOGGER.info("Skipping %s completed %s", skipped, label)
    LOGGER.info("Workers: %s", workers)

    processed = 0
    errors: list[tuple[str, BaseException]] = []

    def record_result(sid: str, result: T) -> None:
        nonlocal processed
        completed[sid] = result
        processed += 1
        if processed % SAVE_EVERY == 0:
            save_completed(completed)
            LOGGER.info("  saved %s/%s %s", len(completed), len(items), label)
        if processed % LOG_EVERY == 0 or processed == len(pending):
            LOGGER.info("  progress %s/%s pending", processed, len(pending))

    if workers == 1:
        for item in pending:
            sid = str(item_id(item))
            try:
                _, result = process_one(item)
                record_result(sid, result)
            except Exception as exc:
                errors.append((sid, exc))
                LOGGER.error("  failed %s %s: %s", label, sid, exc)
                save_completed(completed)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fut_to_sid = {
                executor.submit(process_one, item): str(item_id(item))
                for item in pending
            }
            for future in as_completed(fut_to_sid):
                sid = fut_to_sid[future]
                try:
                    _, result = future.result()
                    record_result(sid, result)
                except Exception as exc:
                    errors.append((sid, exc))
                    LOGGER.error("  failed %s %s: %s", label, sid, exc)
                    save_completed(completed)

    save_completed(completed)
    if errors:
        preview = "; ".join(
            f"{sid}: {type(exc).__name__}: {exc}"
            for sid, exc in errors[:5]
        )
        suffix = " ..." if len(errors) > 5 else ""
        raise RuntimeError(
            f"{len(errors)} {label} failed. Saved {len(completed)}/"
            f"{len(items)} completed {label}. First errors: {preview}{suffix}"
        )
    return completed
