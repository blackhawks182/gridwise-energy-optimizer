from __future__ import annotations

import os

import pytest

from gridwise.models import OptimizeRequest
from gridwise.llm import LLMError, interpret_operator_notes
from gridwise.validator import validate_directives


pytestmark = pytest.mark.skipif(
    os.getenv("GRIDWISE_LIVE_LLM", "0") != "1",
    reason="Set GRIDWISE_LIVE_LLM=1 to run live provider interpretation tests.",
)


def make_request(notes: list[str]) -> OptimizeRequest:
    return OptimizeRequest.model_validate(
        {
            "scenario_id": "LIVE-LLM-001",
            "operator_notes": notes,
            "hours": [
                {
                    "hour": h,
                    "demand_kwh": 100,
                    "solar_kwh": 40,
                    "tariff_bdt_per_kwh": 8,
                }
                for h in range(24)
            ],
            "battery": {
                "capacity_kwh": 200,
                "initial_energy_kwh": 100,
                "minimum_energy_kwh": 20,
                "max_charge_kwh_per_hour": 50,
                "max_discharge_kwh_per_hour": 50,
            },
        }
    )


CASES = [
    (
        [
            "Panel washing from 1 PM until 3 PM will leave only one-fifth of normal rooftop solar.",
            "Keep at least 120 kWh stored from 6 PM until 9 PM.",
            "The sports office changed next month's registration deadline.",
        ],
        [
            (True, "solar_reduction", {"hours": [13, 14], "factor": 0.2}),
            (True, "minimum_battery_reserve", {"hours": [18, 19, 20], "minimum_energy_kwh": 120}),
            (False, "no_op", None),
        ],
    ),
    (
        [
            "No battery charging is permitted between two and four in the afternoon.",
            "From 18:00 through 20:00, battery discharge must be disabled.",
            "Grid import should stay below 50 kWh from 7 PM to 9 PM.",
        ],
        [
            (True, "no_charge_window", {"hours": [14, 15]}),
            (True, "no_discharge_window", {"hours": [18, 19]}),
            (True, "max_grid_window", {"hours": [19, 20], "max_grid_kwh": 50}),
        ],
    ),
    (
        [
            "Expect an eighty percent cut to PV output from noon to 2 PM.",
            "The battery must stay no lower than 80 kWh during 10 PM to midnight.",
            "The library will close early tomorrow.",
        ],
        [
            (True, "solar_reduction", {"hours": [12, 13], "factor": 0.2}),
            (True, "minimum_battery_reserve", {"hours": [22, 23], "minimum_energy_kwh": 80}),
            (False, "no_op", None),
        ],
    ),
    (
        [
            "Rooftop generation is reduced by 25 percent during the 9 AM to 11 AM maintenance period.",
            "Do not import more than 60 kWh between 5 PM and 6 PM.",
            "Battery charging is unavailable from 11 PM until midnight.",
        ],
        [
            (True, "solar_reduction", {"hours": [9, 10], "factor": 0.75}),
            (True, "max_grid_window", {"hours": [17], "max_grid_kwh": 60}),
            (True, "no_charge_window", {"hours": [23]}),
        ],
    ),
]


def test_live_llm_paraphrases():
    for notes, expected in CASES:
        request = make_request(notes)
        try:
            raw = interpret_operator_notes(
                request.operator_notes,
                validator=lambda payload: validate_directives(request, payload),
            )
        except LLMError as exc:
            pytest.fail(f"Live LLM provider failed: {exc}")

        directives = validate_directives(request, raw)
        assert len(directives) == len(expected)

        for item, (applies, directive_type, adjustment) in zip(directives, expected):
            assert item.applies is applies
            assert item.directive_type == directive_type
            if adjustment is None:
                assert item.structured_adjustment is None
            else:
                assert item.structured_adjustment == adjustment
