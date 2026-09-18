from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gridwise.models import OptimizeRequest
from gridwise.optimizer import calculate_totals, format_hourly_plan, optimize_energy
from gridwise.validator import validate_directives, validate_schedule


PUBLIC_CASES_ENV = "GRIDWISE_PUBLIC_CASES"


def test_public_sample_cases_when_supplied():
    path_value = os.getenv(PUBLIC_CASES_ENV)
    if not path_value:
        pytest.skip(f"Set {PUBLIC_CASES_ENV} to the organizer public sample JSON to run this test.")

    path = Path(path_value)
    if not path.is_file():
        pytest.fail(f"Public sample JSON was not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    assert len(cases) == 10

    for case in cases:
        request = OptimizeRequest.model_validate(case["input"])
        directives = validate_directives(
            request,
            {"directive_interpretation": case["expected_output"]["directive_interpretation"]},
        )
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

        expected_cost = float(case["expected_output"]["total_cost_bdt"])
        assert abs(total_cost - expected_cost) <= 0.01, case["id"]
