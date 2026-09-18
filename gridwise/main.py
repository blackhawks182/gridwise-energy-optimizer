from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, HTTPException

from .llm import LLMError, interpret_operator_notes
from .models import HealthResponse, OptimizeRequest, OptimizeResponse
from .optimizer import calculate_totals, format_hourly_plan, optimize_energy
from .validator import validate_directives, validate_schedule


app = FastAPI(
    title="GridWise Energy Optimizer",
    version="1.3.0",
    description=(
        "LLM-assisted GridWise operator directive interpretation, deterministic guardrails, "
        "and 24-hour energy optimization."
    ),
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy_endpoint(request: OptimizeRequest) -> OptimizeResponse:
    try:
        # 1) LLM directly interprets the natural-language operator notes.
        raw_interpretation = interpret_operator_notes(
            request.operator_notes,
            validator=lambda payload: validate_directives(request, payload),
        )

        # 2) Untrusted LLM output is deterministically validated before use.
        directives = validate_directives(request, raw_interpretation)

        # 3) Validated directives become hard constraints in the LP.
        result = optimize_energy(
            request.hours,
            request.battery,
            directives,
        )

        # 4) Convert private solver variables into the public challenge schema.
        hourly_plan = format_hourly_plan(request.hours, result)
        total_grid_kwh, total_cost_bdt, peak_grid_kwh = calculate_totals(
            request.hours,
            hourly_plan,
        )

        # 5) Validate the exact public plan and the exact reported totals.
        validate_schedule(
            request,
            directives,
            hourly_plan,
            total_grid_kwh,
            total_cost_bdt,
            peak_grid_kwh,
        )

        applicable = sum(1 for item in directives if item.applies)
        idle_hours = sum(1 for item in hourly_plan if item["battery_action"] == "idle")
        charge_hours = sum(1 for item in hourly_plan if item["battery_action"] == "charge")
        discharge_hours = sum(1 for item in hourly_plan if item["battery_action"] == "discharge")

        summary = (
            f"Applied {applicable} applicable operator directive(s). "
            "The optimized 24-hour plan minimizes grid electricity cost while satisfying "
            "the battery, solar, energy-balance, end-of-day, and directive constraints. "
            f"Battery schedule: {charge_hours} charge hour(s), {discharge_hours} discharge hour(s), "
            f"and {idle_hours} idle hour(s)."
        )

        return OptimizeResponse(
            scenario_id=request.scenario_id,
            directive_interpretation=directives,
            hourly_plan=hourly_plan,
            total_grid_kwh=round(total_grid_kwh, 6),
            total_cost_bdt=round(total_cost_bdt, 6),
            peak_grid_kwh=round(peak_grid_kwh, 6),
            plan_summary=summary,
        )

    except LLMError as exc:
        # Never expose provider details, prompts, credentials, or stack traces.
        raise HTTPException(
            status_code=500,
            detail="Operator-note interpretation service is unavailable",
        ) from exc
    except ValueError as exc:
        # Covers deterministic guardrail and schedule-validation failures.
        raise HTTPException(
            status_code=500,
            detail="The scenario could not be converted into a valid optimization plan",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail="The optimization service could not produce a valid plan",
        ) from exc


def run_server() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("gridwise.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run_server()
