from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main


PUBLIC_CASES_ENV = "GRIDWISE_PUBLIC_CASES"


def test_public_samples_through_http_boundary(monkeypatch):
    path_value = os.getenv(PUBLIC_CASES_ENV)
    if not path_value:
        pytest.skip(f"Set {PUBLIC_CASES_ENV} to the organizer public sample JSON to run this test.")

    path = Path(path_value)
    if not path.is_file():
        pytest.fail(f"Public sample JSON was not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    assert len(cases) == 10

    expected_by_scenario = {
        case["input"]["scenario_id"]: {
            "directive_interpretation": case["expected_output"]["directive_interpretation"]
        }
        for case in cases
    }

    # TestClient calls the route function in-process, so replace the imported
    # LLM boundary with a request-aware implementation below.
    def route_llm(operator_notes, validator=None):
        # The route receives only notes; map by the full tuple so the test does
        # not bypass the challenge's note-count/order contract.
        for case in cases:
            notes = tuple(case["input"]["operator_notes"])
            if notes == tuple(operator_notes):
                payload = expected_by_scenario[case["input"]["scenario_id"]]
                if validator is not None:
                    validator(payload)
                return payload
        raise AssertionError("Could not map test request to an organizer sample case")

    monkeypatch.setattr(main, "interpret_operator_notes", route_llm)

    client = TestClient(main.app)
    for case in cases:
        response = client.post("/optimize-energy", json=case["input"])
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["scenario_id"] == case["input"]["scenario_id"]
        assert len(payload["directive_interpretation"]) == len(case["input"]["operator_notes"])
        assert len(payload["hourly_plan"]) == 24
        assert abs(
            payload["total_cost_bdt"] - float(case["expected_output"]["total_cost_bdt"])
        ) <= 0.01
