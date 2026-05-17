"""
Shared model API helpers — OpenAI Chat, OpenAI Responses, Anthropic Messages.
"""

import base64
import io
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic
from openai import OpenAI
from PIL import Image

from configs import config
from utils.batch import encode_image_base64, guess_image_media_type, image_data_url

# Claude vision API supported image formats
_ANTHROPIC_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


@dataclass
class ModelClient:
    """Resolved configuration for a model API endpoint."""

    api_model: str
    api_method: str  # "chat" | "responses" | "messages"
    api_key: str
    base_url: str
    max_tokens: Optional[int] = None
    thinking_budget: Optional[int] = None
    openai: Optional[OpenAI] = None
    anthropic: Optional[Anthropic] = None


def create_model_client(model_name: str) -> ModelClient:
    """Resolve model config and return a ModelClient."""
    cfg = config.resolve_model_api_config(model_name)
    mc = ModelClient(
        api_model=cfg["api_model"],
        api_method=cfg["api_method"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        max_tokens=cfg.get("max_tokens"),
        thinking_budget=cfg.get("thinking_budget"),
    )
    if mc.api_method in ("chat", "responses"):
        mc.openai = OpenAI(api_key=mc.api_key, base_url=mc.base_url)
    elif mc.api_method == "messages":
        mc.anthropic = Anthropic(api_key=mc.api_key, base_url=mc.base_url)
    return mc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_vision_model(
    *,
    client: ModelClient,
    prompt: str,
    image_path: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
    require_image: bool = True,
    max_tokens: Optional[int] = None,
) -> str:
    """Call a vision model and return the response text."""
    if image_path and not os.path.exists(image_path):
        if require_image:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_path = None
    elif not image_path and require_image:
        raise FileNotFoundError("Image file path is required")

    if client.api_method == "chat":
        return _call_chat(client, prompt, image_path, system_prompt,
                          temperature, timeout, max_tokens)

    if client.api_method == "responses":
        return _call_responses(client, prompt, image_path, system_prompt,
                               temperature, timeout, max_tokens)

    if client.api_method == "messages":
        return _call_messages(client, prompt, image_path, system_prompt,
                              temperature, timeout, max_tokens)

    raise ValueError(f"Unknown api_method: {client.api_method!r}")


def call_vision_model_with_retries(
    *,
    client: ModelClient,
    prompt: str,
    image_path: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
    require_image: bool = True,
    max_tokens: Optional[int] = None,
    retries: int = 3,
) -> str:
    """Call a vision model with exponential backoff retry."""
    import openai
    last_error = None
    for attempt in range(retries):
        try:
            return call_vision_model(
                client=client, prompt=prompt, image_path=image_path,
                system_prompt=system_prompt, temperature=temperature,
                timeout=timeout, require_image=require_image,
                max_tokens=max_tokens,
            )
        except (openai.APITimeoutError, TimeoutError) as exc:
            # Don't retry timeouts — the server is unresponsive for this sample
            raise RuntimeError(f"API call timed out after {timeout}s: {exc}") from exc
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
    raise RuntimeError(f"API call failed after {retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------


def _call_chat(client: ModelClient, prompt: str, image_path: Optional[str],
               system_prompt: Optional[str], temperature: float,
               timeout: Optional[float], max_tokens: Optional[int]) -> str:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_path:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url(image_path), "detail": "auto"},
        })
    messages.append({"role": "user", "content": content})

    kwargs: dict = {"model": client.api_model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    if client.thinking_budget is not None:
        kwargs["extra_body"] = {
            "thinking": {"budget_tokens": client.thinking_budget},
        }

    response = client.openai.chat.completions.create(**kwargs)
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    msg = choice.message

    # Primary: use content field
    text = (msg.content or "").strip()

    # Fallback: reasoning_content (SiliconFlow Qwen thinking mode)
    if not text:
        text = (getattr(msg, "reasoning_content", None) or "").strip()

    if not text:
        raise ValueError(
            f"API returned empty response content (finish_reason={finish_reason})"
        )
    return text


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


def _call_responses(client: ModelClient, prompt: str, image_path: Optional[str],
                    system_prompt: Optional[str], temperature: float,
                    timeout: Optional[float], max_tokens: Optional[int]) -> str:
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    if image_path:
        content.append({"type": "input_image", "image_url": image_data_url(image_path)})

    kwargs: dict = {
        "model": client.api_model,
        "input": [{"role": "user", "content": content}],
        "temperature": temperature,
    }
    if system_prompt:
        kwargs["instructions"] = system_prompt
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout

    response = client.openai.responses.create(**kwargs)
    text = (getattr(response, "output_text", "") or "").strip()
    if not text:
        raise ValueError("API returned empty response content")
    return text


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------


def _call_messages(client: ModelClient, prompt: str, image_path: Optional[str],
                   system_prompt: Optional[str], temperature: float,
                   timeout: Optional[float], max_tokens: Optional[int]) -> str:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_path:
        content.append(_build_anthropic_image_block(image_path))

    kwargs: dict = {
        "model": client.api_model,
        "max_tokens": max_tokens or 4096,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if timeout is not None:
        kwargs["timeout"] = timeout

    with client.anthropic.messages.stream(**kwargs) as stream:
        response = stream.get_final_message()
    text_parts = [block.text for block in response.content if block.type == "text"]
    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError("API returned empty response content")
    return text


def _build_anthropic_image_block(image_path: str) -> dict:
    """Build an Anthropic image content block, converting unsupported formats."""
    media_type = guess_image_media_type(image_path)

    if media_type in _ANTHROPIC_IMAGE_TYPES:
        image_data = encode_image_base64(image_path)
    else:
        with Image.open(image_path) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG")
            image_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        media_type = "image/jpeg"

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": image_data,
        },
    }
