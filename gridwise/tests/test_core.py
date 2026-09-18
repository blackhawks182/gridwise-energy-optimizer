from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from gridwise.main import app
from gridwise.models import OptimizeRequest
from gridwise.optimizer import calculate_totals, format_hourly_plan, optimize_energy
from gridwise.validator import validate_directives, validate_schedule


def make_request() -> OptimizeRequest:
    return OptimizeRequest.model_validate(
        {
            "scenario_id": "TEST-001",
            "operator_notes": [
                "Test note one",
                "Test note two",
                "Test note three",
            ],
            "hours": [
                {
                    "hour": h,
                    "demand_kwh": 10,
                    "solar_kwh": 5,
                    "tariff_bdt_per_kwh": 10 + (h % 3),
                }
                for h in range(24)
            ],
            "battery": {
                "capacity_kwh": 20,
                "initial_energy_kwh": 10,
                "minimum_energy_kwh": 2,
                "max_charge_kwh_per_hour": 5,
                "max_discharge_kwh_per_hour": 5,
            },
        }
    )


def all_five_directives():
    return {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": [6, 7], "factor": 0.2},
                "explanation": "Solar is reduced in the test window.",
            },
            {
                "note_index": 1,
                "applies": True,
                "directive_type": "minimum_battery_reserve",
                "structured_adjustment": {
                    "hours": [8, 9],
                    "minimum_energy_kwh": 8,
                },
                "explanation": "Battery reserve is raised in the test window.",
            },
            {
                "note_index": 2,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": [10, 11]},
                "explanation": "Charging is unavailable in the test window.",
            },
        ]
    }


def add_two_directives(raw):
    raw["directive_interpretation"].append(
        {
            "note_index": 3,
            "applies": True,
            "directive_type": "no_discharge_window",
            "structured_adjustment": {"hours": [12, 13]},
            "explanation": "Discharging is unavailable in the test window.",
        }
    )
    return raw


def test_models_reject_nonfinite_values():
    with pytest.raises(Exception):
        OptimizeRequest.model_validate(
            {
                "scenario_id": "NONFINITE",
                "operator_notes": ["x"],
                "hours": [
                    {
                        "hour": h,
                        "demand_kwh": math.inf if h == 0 else 1,
                        "solar_kwh": 0,
                        "tariff_bdt_per_kwh": 1,
                    }
                    for h in range(24)
                ],
                "battery": {
                    "capacity_kwh": 10,
                    "initial_energy_kwh": 5,
                    "minimum_energy_kwh": 1,
                    "max_charge_kwh_per_hour": 1,
                    "max_discharge_kwh_per_hour": 1,
                },
            }
        )


def test_optimizer_and_schedule_validator():
    request = make_request()
    raw = all_five_directives()
    directives = validate_directives(request, raw)

    result = optimize_energy(request.hours, request.battery, directives)
    plan = format_hourly_plan(request.hours, result)
    total_grid, total_cost, peak_grid = calculate_totals(request.hours, plan)

    validate_schedule(
        request,
        directives,
        plan,
        total_grid,
        total_cost,
        peak_grid,
    )

    assert len(plan) == 24
    assert plan[6]["solar_used_kwh"] <= 1.01
    assert plan[7]["solar_used_kwh"] <= 1.01
    assert plan[8]["battery_energy_after_kwh"] >= 8 - 0.01
    assert plan[9]["battery_energy_after_kwh"] >= 8 - 0.01
    assert plan[10]["battery_action"] != "charge"
    assert plan[11]["battery_action"] != "charge"
    assert abs(plan[-1]["battery_energy_after_kwh"] - 10) <= 0.01


def test_no_discharge_window_is_enforced_with_a_better_targeted_case():
    request = make_request()
    directives = validate_directives(
        request,
        {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_discharge_window",
                    "structured_adjustment": {"hours": list(range(2, 8))},
                    "explanation": "No battery discharge in the test window.",
                },
                {
                    "note_index": 1,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "No effect.",
                },
                {
                    "note_index": 2,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "No effect.",
                },
            ]
        },
    )
    result = optimize_energy(request.hours, request.battery, directives)
    plan = format_hourly_plan(request.hours, result)
    for h in range(2, 8):
        assert plan[h]["battery_action"] != "discharge"


def test_max_grid_window_is_enforced():
    request = make_request()
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": [10, 11], "max_grid_kwh": 6},
                "explanation": "Grid import is capped in the test window.",
            },
            {
                "note_index": 1,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No schedule effect.",
            },
            {
                "note_index": 2,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No schedule effect.",
            },
        ]
    }
    directives = validate_directives(request, raw)
    result = optimize_energy(request.hours, request.battery, directives)
    plan = format_hourly_plan(request.hours, result)

    assert plan[10]["grid_kwh"] <= 6.01
    assert plan[11]["grid_kwh"] <= 6.01


def test_validator_catches_tampering():
    request = make_request()
    raw = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No schedule effect.",
            },
            {
                "note_index": 1,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No schedule effect.",
            },
            {
                "note_index": 2,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No schedule effect.",
            },
        ]
    }
    directives = validate_directives(request, raw)
    result = optimize_energy(request.hours, request.battery, directives)
    plan = format_hourly_plan(request.hours, result)
    total_grid, total_cost, peak_grid = calculate_totals(request.hours, plan)

    plan[0]["grid_kwh"] += 0.02
    with pytest.raises(ValueError, match="energy-balance violation"):
        validate_schedule(request, directives, plan, total_grid, total_cost, peak_grid)


def test_directive_guardrails_reject_bad_interpretations():
    request = make_request()

    bad_factor = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": [1], "factor": 1.2},
                "explanation": "Bad factor.",
            },
            {
                "note_index": 1,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No effect.",
            },
            {
                "note_index": 2,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No effect.",
            },
        ]
    }
    with pytest.raises(ValueError):
        validate_directives(request, bad_factor)

    unsorted_hours = {
        "directive_interpretation": [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": [4, 3]},
                "explanation": "Unsorted window.",
            },
            {
                "note_index": 1,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No effect.",
            },
            {
                "note_index": 2,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No effect.",
            },
        ]
    }
    with pytest.raises(ValueError):
        validate_directives(request, unsorted_hours)


def test_api_with_mocked_llm(monkeypatch):
    request = make_request()
    raw = all_five_directives()

    monkeypatch.setattr("gridwise.main.interpret_operator_notes", lambda notes, validator=None: raw)

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    response = client.post("/optimize-energy", json=request.model_dump())
    assert response.status_code == 200
    payload = response.json()
    assert payload["scenario_id"] == "TEST-001"
    assert len(payload["directive_interpretation"]) == 3
    assert len(payload["hourly_plan"]) == 24
    assert abs(payload["hourly_plan"][-1]["battery_energy_after_kwh"] - 10) <= 0.01
    assert payload["total_cost_bdt"] >= 0
