from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog


ZERO_TOLERANCE = 1e-7


def _directive_type(directive: Any) -> str:
    if isinstance(directive, dict):
        return directive["directive_type"]
    return directive.directive_type


def _adjustment(directive: Any) -> dict[str, Any] | None:
    if isinstance(directive, dict):
        return directive.get("structured_adjustment")
    return directive.structured_adjustment


def _get_value(obj: Any, key: str) -> float:
    if isinstance(obj, dict):
        return float(obj[key])
    return float(getattr(obj, key))


def _effective_solar(hours: list[Any], directives: list[Any], hour: int) -> float:
    available = _get_value(hours[hour], "solar_kwh")

    for directive in directives:
        if _directive_type(directive) != "solar_reduction":
            continue
        adjustment = _adjustment(directive)
        if adjustment and hour in adjustment["hours"]:
            available *= float(adjustment["factor"])

    return available


def _minimum_energy_for_hour(
    battery: Any,
    directives: list[Any],
    hour: int,
) -> float:
    minimum_energy = _get_value(battery, "minimum_energy_kwh")

    for directive in directives:
        if _directive_type(directive) != "minimum_battery_reserve":
            continue
        adjustment = _adjustment(directive)
        if adjustment and hour in adjustment["hours"]:
            minimum_energy = max(
                minimum_energy,
                float(adjustment["minimum_energy_kwh"]),
            )

    return minimum_energy


def optimize_energy(
    hours: list[Any],
    battery: Any,
    directives: list[Any] | None = None,
):
    """
    Solve the 24-hour GridWise linear program.

    Internal variables per hour:
      grid[h]             >= 0
      solar_used[h]       >= 0
      battery_action[h]   signed: + charge, - discharge
      battery_energy[h]   state after the hour

    A single signed battery-action variable prevents simultaneous charging
    and discharging by construction.
    """
    directives = directives or []
    n = len(hours)
    if n != 24:
        raise ValueError("GridWise optimization requires exactly 24 hours")

    grid = list(range(n))
    solar_used = list(range(n, 2 * n))
    battery_action = list(range(2 * n, 3 * n))
    battery_energy = list(range(3 * n, 4 * n))
    total_variables = 4 * n

    objective = np.zeros(total_variables, dtype=float)
    for h in range(n):
        objective[grid[h]] = _get_value(hours[h], "tariff_bdt_per_kwh")

    equality_constraints: list[np.ndarray] = []
    equality_values: list[float] = []

    # grid + solar_used - battery_action = demand
    for h in range(n):
        row = np.zeros(total_variables, dtype=float)
        row[grid[h]] = 1.0
        row[solar_used[h]] = 1.0
        row[battery_action[h]] = -1.0
        equality_constraints.append(row)
        equality_values.append(_get_value(hours[h], "demand_kwh"))

    # battery_energy[0] - battery_action[0] = initial_energy
    row = np.zeros(total_variables, dtype=float)
    row[battery_energy[0]] = 1.0
    row[battery_action[0]] = -1.0
    equality_constraints.append(row)
    equality_values.append(_get_value(battery, "initial_energy_kwh"))

    # battery_energy[h] - battery_energy[h-1] - battery_action[h] = 0
    for h in range(1, n):
        row = np.zeros(total_variables, dtype=float)
        row[battery_energy[h]] = 1.0
        row[battery_energy[h - 1]] = -1.0
        row[battery_action[h]] = -1.0
        equality_constraints.append(row)
        equality_values.append(0.0)

    # Final state of charge must equal the initial state.
    row = np.zeros(total_variables, dtype=float)
    row[battery_energy[n - 1]] = 1.0
    equality_constraints.append(row)
    equality_values.append(_get_value(battery, "initial_energy_kwh"))

    inequality_constraints: list[np.ndarray] = []
    inequality_values: list[float] = []

    for directive in directives:
        directive_type = _directive_type(directive)
        adjustment = _adjustment(directive)
        if not adjustment:
            continue

        if directive_type == "no_charge_window":
            # battery_action[h] <= 0
            for h in adjustment["hours"]:
                row = np.zeros(total_variables, dtype=float)
                row[battery_action[h]] = 1.0
                inequality_constraints.append(row)
                inequality_values.append(0.0)

        elif directive_type == "no_discharge_window":
            # -battery_action[h] <= 0  <=>  battery_action[h] >= 0
            for h in adjustment["hours"]:
                row = np.zeros(total_variables, dtype=float)
                row[battery_action[h]] = -1.0
                inequality_constraints.append(row)
                inequality_values.append(0.0)

        elif directive_type == "max_grid_window":
            for h in adjustment["hours"]:
                row = np.zeros(total_variables, dtype=float)
                row[grid[h]] = 1.0
                inequality_constraints.append(row)
                inequality_values.append(float(adjustment["max_grid_kwh"]))

    bounds: list[tuple[float, float | None]] = []

    # Grid import only; negative grid would be export, which is not allowed.
    bounds.extend((0.0, None) for _ in range(n))

    # Solar can be curtailed; it cannot exceed effective usable solar.
    for h in range(n):
        bounds.append((0.0, _effective_solar(hours, directives, h)))

    max_charge = _get_value(battery, "max_charge_kwh_per_hour")
    max_discharge = _get_value(battery, "max_discharge_kwh_per_hour")
    for _ in range(n):
        bounds.append((-max_discharge, max_charge))

    capacity = _get_value(battery, "capacity_kwh")
    for h in range(n):
        bounds.append((_minimum_energy_for_hour(battery, directives, h), capacity))

    result = linprog(
        c=objective,
        A_eq=np.asarray(equality_constraints, dtype=float),
        b_eq=np.asarray(equality_values, dtype=float),
        A_ub=np.asarray(inequality_constraints, dtype=float)
        if inequality_constraints
        else None,
        b_ub=np.asarray(inequality_values, dtype=float)
        if inequality_values
        else None,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError("Optimization could not find a feasible schedule")

    if result.x is None or not np.all(np.isfinite(result.x)):
        raise RuntimeError("Optimization returned invalid numeric values")

    return result


def _clean_nonnegative(value: float) -> float:
    value = float(value)
    if abs(value) < ZERO_TOLERANCE:
        return 0.0
    return round(max(0.0, value), 6)


def format_hourly_plan(hours: list[Any], result) -> list[dict[str, Any]]:
    """Convert the internal LP variables into the challenge's public plan schema."""
    n = len(hours)
    if n != 24:
        raise ValueError("GridWise hourly_plan requires exactly 24 hours")

    grid = result.x[0:n]
    solar_used = result.x[n:2 * n]
    signed_action = result.x[2 * n:3 * n]
    battery_energy = result.x[3 * n:4 * n]

    plan: list[dict[str, Any]] = []

    for h in range(n):
        action = float(signed_action[h])
        if abs(action) < ZERO_TOLERANCE:
            action_name = "idle"
            battery_kwh = 0.0
        elif action > 0:
            action_name = "charge"
            battery_kwh = action
        else:
            action_name = "discharge"
            battery_kwh = -action

        plan.append(
            {
                "hour": h,
                "grid_kwh": _clean_nonnegative(grid[h]),
                "solar_used_kwh": _clean_nonnegative(solar_used[h]),
                "battery_action": action_name,
                "battery_kwh": _clean_nonnegative(battery_kwh),
                "battery_energy_after_kwh": _clean_nonnegative(battery_energy[h]),
            }
        )

    return plan


def calculate_totals(
    hours: list[Any],
    hourly_plan: list[dict[str, Any]],
) -> tuple[float, float, float]:
    total_grid = sum(float(item["grid_kwh"]) for item in hourly_plan)
    total_cost = sum(
        float(item["grid_kwh"])
        * _get_value(hours[item["hour"]], "tariff_bdt_per_kwh")
        for item in hourly_plan
    )
    peak_grid = max(float(item["grid_kwh"]) for item in hourly_plan)
    return total_grid, total_cost, peak_grid
