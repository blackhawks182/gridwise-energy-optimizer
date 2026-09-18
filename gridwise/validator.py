from __future__ import annotations

import math
from typing import Any

from pydantic import ValidationError

from .models import (
    DirectiveInterpretation,
    HourlyPlanEntry,
    LLMInterpretationEnvelope,
    MaxGridWindowAdjustment,
    MinimumBatteryReserveAdjustment,
    OptimizeRequest,
    SolarReductionAdjustment,
    WindowAdjustment,
)


TOLERANCE = 0.01
ALLOWED_DIRECTIVES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}
WINDOW_DIRECTIVES = {"no_charge_window", "no_discharge_window"}


def _value(obj: Any, key: str) -> float:
    if isinstance(obj, dict):
        return float(obj[key])
    return float(getattr(obj, key))


def _validate_hours(hours: Any) -> list[int]:
    if not isinstance(hours, list):
        raise ValueError("directive hours must be a list")
    if any(not isinstance(hour, int) or isinstance(hour, bool) for hour in hours):
        raise ValueError("directive hours must contain only integers")
    if len(hours) != len(set(hours)):
        raise ValueError("directive hours must be unique")
    if any(hour < 0 or hour > 23 for hour in hours):
        raise ValueError("directive hours must be between 0 and 23")
    if hours != sorted(hours):
        raise ValueError("directive hours must be in ascending order")
    return hours


def validate_directives(
    request: OptimizeRequest,
    raw_payload: Any,
) -> list[DirectiveInterpretation]:
    """Deterministically validate and normalize untrusted LLM output."""
    try:
        envelope = LLMInterpretationEnvelope.model_validate(raw_payload)
    except ValidationError as exc:
        raise ValueError("LLM interpretation did not match the required schema") from exc

    items = envelope.directive_interpretation
    if len(items) != len(request.operator_notes):
        raise ValueError("LLM must return exactly one interpretation per operator note")

    validated: list[DirectiveInterpretation] = []

    for expected_index, item in enumerate(items):
        if item.note_index != expected_index:
            raise ValueError("note_index values must exactly match note order")
        if item.directive_type not in ALLOWED_DIRECTIVES:
            raise ValueError("unsupported directive_type")
        if not item.explanation.strip():
            raise ValueError("directive explanation cannot be empty")

        adjustment = item.structured_adjustment
        directive_type = item.directive_type

        if directive_type == "no_op":
            if item.applies is not False or adjustment is not None:
                raise ValueError("no_op requires applies=false and structured_adjustment=null")
            validated.append(item)
            continue

        if item.applies is not True:
            raise ValueError("every non-no_op directive requires applies=true")
        if adjustment is None:
            raise ValueError("non-no_op directives require structured_adjustment")

        try:
            if directive_type == "solar_reduction":
                parsed = SolarReductionAdjustment.model_validate(adjustment)
                hours = _validate_hours(parsed.hours)
                factor = float(parsed.factor)
                if not math.isfinite(factor) or not 0 <= factor <= 1:
                    raise ValueError("solar_reduction factor must be finite and between 0 and 1")
                normalized = parsed.model_copy(
                    update={"hours": hours, "factor": factor}
                ).model_dump()

            elif directive_type == "minimum_battery_reserve":
                parsed = MinimumBatteryReserveAdjustment.model_validate(adjustment)
                hours = _validate_hours(parsed.hours)
                reserve = float(parsed.minimum_energy_kwh)
                capacity = _value(request.battery, "capacity_kwh")
                if not math.isfinite(reserve) or reserve < 0 or reserve > capacity:
                    raise ValueError(
                        "minimum battery reserve must be finite, non-negative, and <= capacity"
                    )
                normalized = parsed.model_copy(
                    update={"hours": hours, "minimum_energy_kwh": reserve}
                ).model_dump()

            elif directive_type in WINDOW_DIRECTIVES:
                parsed = WindowAdjustment.model_validate(adjustment)
                normalized = parsed.model_copy(
                    update={"hours": _validate_hours(parsed.hours)}
                ).model_dump()

            elif directive_type == "max_grid_window":
                parsed = MaxGridWindowAdjustment.model_validate(adjustment)
                hours = _validate_hours(parsed.hours)
                max_grid = float(parsed.max_grid_kwh)
                if not math.isfinite(max_grid) or max_grid < 0:
                    raise ValueError("max_grid_kwh must be finite and non-negative")
                normalized = parsed.model_copy(
                    update={"hours": hours, "max_grid_kwh": max_grid}
                ).model_dump()

            else:
                raise ValueError("unsupported directive_type")
        except ValidationError as exc:
            raise ValueError("directive structured_adjustment has the wrong shape") from exc

        validated.append(item.model_copy(update={"structured_adjustment": normalized}))

    return validated


def _effective_solar_for_request(
    request: OptimizeRequest,
    directives: list[DirectiveInterpretation],
) -> list[float]:
    effective_solar = [float(hour.solar_kwh) for hour in request.hours]
    for directive in directives:
        if directive.directive_type != "solar_reduction":
            continue
        adjustment = directive.structured_adjustment or {}
        for h in adjustment["hours"]:
            effective_solar[h] *= float(adjustment["factor"])
    return effective_solar


def validate_schedule(
    request: OptimizeRequest,
    directives: list[DirectiveInterpretation],
    hourly_plan: list[dict[str, Any]],
    total_grid_kwh: float,
    total_cost_bdt: float,
    peak_grid_kwh: float,
) -> None:
    """Independently validate the final public schedule and reported totals."""
    if len(hourly_plan) != 24:
        raise ValueError("hourly_plan must contain exactly 24 entries")

    try:
        entries = [HourlyPlanEntry.model_validate(item) for item in hourly_plan]
    except ValidationError as exc:
        raise ValueError("hourly_plan contains invalid entries") from exc

    hour_numbers = [entry.hour for entry in entries]
    if sorted(hour_numbers) != list(range(24)) or len(set(hour_numbers)) != 24:
        raise ValueError("hourly_plan must contain unique hours 0 through 23")

    hour_map = {entry.hour: entry for entry in entries}

    previous_energy = _value(request.battery, "initial_energy_kwh")
    base_minimum = _value(request.battery, "minimum_energy_kwh")
    capacity = _value(request.battery, "capacity_kwh")
    max_charge = _value(request.battery, "max_charge_kwh_per_hour")
    max_discharge = _value(request.battery, "max_discharge_kwh_per_hour")
    effective_solar = _effective_solar_for_request(request, directives)

    for h in range(24):
        item = hour_map[h]
        grid = float(item.grid_kwh)
        solar = float(item.solar_used_kwh)
        battery_kwh = float(item.battery_kwh)
        energy_after = float(item.battery_energy_after_kwh)
        demand = float(request.hours[h].demand_kwh)

        if grid < -TOLERANCE:
            raise ValueError(f"negative grid usage at hour {h}")
        if solar < -TOLERANCE:
            raise ValueError(f"negative solar usage at hour {h}")
        if solar > effective_solar[h] + TOLERANCE:
            raise ValueError(f"solar usage exceeds effective solar at hour {h}")

        if item.battery_action == "charge":
            signed_action = battery_kwh
        elif item.battery_action == "discharge":
            signed_action = -battery_kwh
        elif item.battery_action == "idle":
            signed_action = 0.0
            if abs(battery_kwh) > TOLERANCE:
                raise ValueError(f"idle battery_action must have battery_kwh=0 at hour {h}")
        else:
            raise ValueError(f"unsupported battery_action at hour {h}")

        if signed_action > max_charge + TOLERANCE:
            raise ValueError(f"charge-rate violation at hour {h}")
        if signed_action < -max_discharge - TOLERANCE:
            raise ValueError(f"discharge-rate violation at hour {h}")

        expected_energy = previous_energy + signed_action
        if abs(energy_after - expected_energy) > TOLERANCE:
            raise ValueError(f"battery energy transition violation at hour {h}")

        active_minimum = base_minimum
        for directive in directives:
            if directive.directive_type != "minimum_battery_reserve":
                continue
            adjustment = directive.structured_adjustment or {}
            if h in adjustment["hours"]:
                active_minimum = max(
                    active_minimum,
                    float(adjustment["minimum_energy_kwh"]),
                )

        if energy_after < active_minimum - TOLERANCE:
            raise ValueError(f"battery minimum-energy violation at hour {h}")
        if energy_after > capacity + TOLERANCE:
            raise ValueError(f"battery capacity violation at hour {h}")

        # Public-contract energy balance:
        # grid + solar + discharge = demand + charge
        if abs(grid + solar - signed_action - demand) > TOLERANCE:
            raise ValueError(f"energy-balance violation at hour {h}")

        for directive in directives:
            adjustment = directive.structured_adjustment
            if adjustment is None or h not in adjustment["hours"]:
                continue
            if directive.directive_type == "no_charge_window" and signed_action > TOLERANCE:
                raise ValueError(f"no_charge_window violation at hour {h}")
            if directive.directive_type == "no_discharge_window" and signed_action < -TOLERANCE:
                raise ValueError(f"no_discharge_window violation at hour {h}")
            if directive.directive_type == "max_grid_window":
                if grid > float(adjustment["max_grid_kwh"]) + TOLERANCE:
                    raise ValueError(f"max_grid_window violation at hour {h}")

        previous_energy = energy_after

    initial_energy = _value(request.battery, "initial_energy_kwh")
    if abs(previous_energy - initial_energy) > TOLERANCE:
        raise ValueError("final battery energy must equal initial energy")

    try:
        total_grid_kwh = float(total_grid_kwh)
        total_cost_bdt = float(total_cost_bdt)
        peak_grid_kwh = float(peak_grid_kwh)
    except (TypeError, ValueError) as exc:
        raise ValueError("reported totals must be numeric") from exc

    if not all(math.isfinite(value) for value in (total_grid_kwh, total_cost_bdt, peak_grid_kwh)):
        raise ValueError("reported totals must be finite")

    recalculated_grid = sum(float(item.grid_kwh) for item in entries)
    recalculated_cost = sum(
        float(item.grid_kwh) * float(request.hours[item.hour].tariff_bdt_per_kwh)
        for item in entries
    )
    recalculated_peak = max(float(item.grid_kwh) for item in entries)

    if abs(recalculated_grid - total_grid_kwh) > TOLERANCE:
        raise ValueError("total_grid_kwh does not match hourly_plan")
    if abs(recalculated_cost - total_cost_bdt) > TOLERANCE:
        raise ValueError("total_cost_bdt does not match hourly_plan")
    if abs(recalculated_peak - peak_grid_kwh) > TOLERANCE:
        raise ValueError("peak_grid_kwh does not match hourly_plan")
