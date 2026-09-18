from __future__ import annotations

import json
import os
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()


class LLMError(RuntimeError):
    """Controlled error for provider/configuration failures."""


SYSTEM_PROMPT = r"""
You are the GridWise operator-note interpreter.

Your ONLY task is to translate each natural-language operator note into one
machine-checkable directive interpretation. Do not optimize the energy plan.
Do not alter demand, solar availability, tariffs, battery parameters, or any
other scenario data. Do not invent unsupported directive types.

SUPPORTED DIRECTIVES AND EXACT ADJUSTMENT SHAPES
- solar_reduction: {"hours":[...],"factor":number}
- minimum_battery_reserve: {"hours":[...],"minimum_energy_kwh":number}
- no_charge_window: {"hours":[...]}
- no_discharge_window: {"hours":[...]}
- max_grid_window: {"hours":[...],"max_grid_kwh":number}
- no_op: null

INTERPRETATION RULES
1. Return exactly one entry for every input note.
2. note_index is zero-based and must match the original note order exactly.
3. Use applies=false ONLY for no_op. Every non-no_op directive uses applies=true.
4. Every hours array contains unique integer hours from 0 through 23 in
   ascending order.
5. Whole-hour windows are start-inclusive and end-exclusive:
   1 PM to 3 PM -> [13,14].
6. Convert 12-hour clock expressions to 24-hour hour numbers. Noon is 12;
   midnight is 0. “until 2 PM” in a window ending at 2 PM means the 2 PM hour
   itself is excluded when the wording describes a start-to-end interval.
7. For solar_reduction, factor is the usable fraction that REMAINS.
   “Only 20% remains” -> 0.2.
   “An 80% reduction” -> 0.2.
   “Reduced by 20%” -> 0.8.
8. Extract numeric limits exactly as stated. Do not estimate, round, or invent.
9. If a note is irrelevant to the current 24-hour energy schedule, use no_op.
10. A note about a real operational topic is still no_op unless it changes one
    of the six supported energy directives.
11. Keep each explanation short, factual, and about the interpretation only.
12. Output valid JSON only. Do not use markdown fences or commentary.

EXAMPLES
Note: "Solar cleaning from noon until 2 PM leaves about 25% of forecast output."
=> solar_reduction; hours [12,13]; factor 0.25

Note: "Do not charge the battery between 2 PM and 4 PM."
=> no_charge_window; hours [14,15]

Note: "Keep at least 120 kWh in reserve from 6 PM until 9 PM."
=> minimum_battery_reserve; hours [18,19,20]; minimum_energy_kwh 120

Note: "The cafeteria menu changes tomorrow."
=> no_op; applies=false; structured_adjustment=null

REQUIRED JSON SHAPE
{
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12,13], "factor": 0.25},
      "explanation": "Usable solar is reduced during the cleaning window."
    }
  ]
}
""".strip()


ProviderValidator = Callable[[dict[str, Any]], Any]


PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_env": "GEMINI_MODEL",
        "model": "gemini-3.8-flash",
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "base_url_env": "GROQ_BASE_URL",
        "base_url": "https://api.groq.com/openai/v1",
        "model_env": "GROQ_MODEL",
        "model": "openai/gpt-oss-120b",
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "base_url": "https://openrouter.ai/api/v1",
        "model_env": "OPENROUTER_MODEL",
        "model": "openrouter/free",
    },
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_MODEL",
        "model": "deepseek-flash",
    },
}

DEFAULT_PROVIDER_ORDER = "gemini,groq,openrouter,deepseek"


def _read_positive_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise LLMError(f"{name} is invalid") from exc
    if value < minimum or value > maximum:
        raise LLMError(f"{name} is outside the supported range")
    return value


def _read_timeout() -> float:
    raw = os.getenv("LLM_TIMEOUT_SECONDS", os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "15")).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise LLMError("LLM_TIMEOUT_SECONDS is invalid") from exc
    if value <= 0:
        raise LLMError("LLM_TIMEOUT_SECONDS must be positive")
    return value


def _provider_order() -> list[str]:
    raw = os.getenv("LLM_PROVIDER_ORDER", DEFAULT_PROVIDER_ORDER)
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not names:
        raise LLMError("LLM_PROVIDER_ORDER is empty")
    unknown = [name for name in names if name not in PROVIDER_DEFAULTS]
    if unknown:
        raise LLMError("LLM_PROVIDER_ORDER contains an unsupported provider")
    return names


def _configured_provider_order() -> list[str]:
    configured: list[str] = []
    for provider in _provider_order():
        key_env = PROVIDER_DEFAULTS[provider]["key_env"]
        if os.getenv(key_env, "").strip():
            configured.append(provider)
    return configured


def _provider_config(provider: str) -> tuple[str, str, str]:
    spec = PROVIDER_DEFAULTS[provider]
    api_key = os.getenv(spec["key_env"], "").strip()
    if not api_key:
        raise LLMError(f"{spec['key_env']} is not configured")

    base_url = os.getenv(spec["base_url_env"], spec["base_url"]).strip()
    model = os.getenv(spec["model_env"], spec["model"]).strip() or spec["model"]
    return api_key, base_url, model


def _make_client(provider: str):
    api_key, base_url, _ = _provider_config(provider)

    try:
        from openai import OpenAI
    except Exception as exc:
        raise LLMError("OpenAI-compatible SDK is not installed") from exc

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=_read_timeout(),
        max_retries=0,
    )


def _json_from_content(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise LLMError("LLM returned empty interpretation content")

    text = content.strip()

    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip("\n ")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError("LLM returned invalid JSON") from exc

    if not isinstance(parsed, dict):
        raise LLMError("LLM returned the wrong JSON top-level shape")

    return parsed


def _call_provider(provider: str, operator_notes: list[str]) -> dict[str, Any]:
    client = _make_client(provider)
    _, _, model = _provider_config(provider)
    max_tokens = _read_positive_int("LLM_MAX_TOKENS", 1400, 256, 8000)

    user_payload = {
        "operator_notes": [
            {"note_index": index, "note": note}
            for index, note in enumerate(operator_notes)
        ],
        "instruction": "Interpret every note and return the required JSON object only.",
    }

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    if provider == "groq" and model.startswith("openai/"):
        kwargs["reasoning_effort"] = os.getenv("GROQ_REASONING_EFFORT", "low")
        kwargs["reasoning_format"] = "hidden"

    if provider == "deepseek":
        thinking_enabled = os.getenv("DEEPSEEK_THINKING", "false").strip().lower() == "true"
        kwargs["extra_body"] = {
            "thinking": {
                "type": "enabled" if thinking_enabled else "disabled"
            }
        }

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise LLMError(f"{provider} interpretation request failed") from exc

    try:
        choices = response.choices
        if not choices:
            raise LLMError(f"{provider} returned no choices")

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise LLMError(f"{provider} interpretation was truncated")

        content = getattr(choice.message, "content", None)
        return _json_from_content(content)
    except LLMError:
        raise
    except (AttributeError, TypeError) as exc:
        raise LLMError(f"{provider} returned an unusable response") from exc


def interpret_operator_notes(
    operator_notes: list[str],
    validator: ProviderValidator | None = None,
) -> dict[str, Any]:
    """
    Interpret notes through the configured provider chain.

    Providers without configured keys are skipped. Provider failures, truncated
    outputs, malformed JSON, and deterministic validation failures can fall
    through to the next configured provider. The caller's validator is invoked
    before a provider result is accepted when supplied.
    """
    last_error: Exception | None = None
    providers = _configured_provider_order()

    for provider in providers:
        try:
            payload = _call_provider(provider, operator_notes)
            if validator is not None:
                validator(payload)
            return payload
        except (LLMError, ValueError) as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise LLMError("No configured LLM provider produced a valid interpretation") from last_error
    raise LLMError("No configured LLM provider keys are available")
