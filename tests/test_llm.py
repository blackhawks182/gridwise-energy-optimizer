from __future__ import annotations

import sys
import types

import pytest

import llm


VALID_PAYLOAD = {
    "directive_interpretation": [
        {
            "note_index": 0,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "No schedule effect.",
        }
    ]
}


class FakeCompletions:
    def __init__(self, payload: str, finish_reason: str = "stop"):
        self.payload = payload
        self.finish_reason = finish_reason
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        choice = types.SimpleNamespace(
            finish_reason=self.finish_reason,
            message=types.SimpleNamespace(content=self.payload),
        )
        return types.SimpleNamespace(choices=[choice])


class FakeClient:
    current = None

    def __init__(self, **kwargs):
        self.config = kwargs
        self.chat = types.SimpleNamespace(completions=FakeClient.current)


def install_fake_openai(monkeypatch, completions: FakeCompletions):
    FakeClient.current = completions
    fake_openai = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)


def test_gemini_request_configuration(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini")

    completions = FakeCompletions(
        '{"directive_interpretation":[{"note_index":0,"applies":false,"directive_type":"no_op","structured_adjustment":null,"explanation":"No schedule effect."}]}'
    )
    install_fake_openai(monkeypatch, completions)

    result = llm.interpret_operator_notes(["The menu changes tomorrow."])
    assert result == VALID_PAYLOAD
    assert completions.kwargs["model"] == "gemini-3.8-flash"
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["temperature"] == 0


def test_deepseek_adds_explicit_thinking_control(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_THINKING", "false")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "deepseek")

    completions = FakeCompletions(
        '{"directive_interpretation":[{"note_index":0,"applies":false,"directive_type":"no_op","structured_adjustment":null,"explanation":"No schedule effect."}]}'
    )
    install_fake_openai(monkeypatch, completions)

    llm.interpret_operator_notes(["The menu changes tomorrow."])
    assert completions.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_llm_rejects_truncated_output(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "deepseek")
    completions = FakeCompletions("{", finish_reason="length")
    install_fake_openai(monkeypatch, completions)

    with pytest.raises(llm.LLMError, match="No configured LLM provider"):
        llm.interpret_operator_notes(["test"])


def test_provider_chain_skips_missing_keys_and_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,groq")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    attempts: list[str] = []

    def fake_call(provider, notes):
        attempts.append(provider)
        return VALID_PAYLOAD

    monkeypatch.setattr(llm, "_call_provider", fake_call)

    result = llm.interpret_operator_notes(["test"])
    assert result == VALID_PAYLOAD
    assert attempts == ["groq"]


def test_provider_chain_falls_back_after_validation_failure(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,groq")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    attempts: list[str] = []

    bad_payload = {"directive_interpretation": []}

    def fake_call(provider, notes):
        attempts.append(provider)
        return bad_payload if provider == "gemini" else VALID_PAYLOAD

    monkeypatch.setattr(llm, "_call_provider", fake_call)

    def reject_first(payload):
        if payload is bad_payload:
            raise ValueError("semantic guardrail rejection")

    result = llm.interpret_operator_notes(["test"], validator=reject_first)
    assert result == VALID_PAYLOAD
    assert attempts == ["gemini", "groq"]


def test_invalid_provider_order_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "not-a-provider")
    with pytest.raises(llm.LLMError, match="unsupported provider"):
        llm.interpret_operator_notes(["test"])


def test_no_provider_keys_is_controlled_failure(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,groq")
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(llm.LLMError, match="No configured LLM provider"):
        llm.interpret_operator_notes(["test"])
