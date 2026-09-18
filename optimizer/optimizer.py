import numpy as np
from scipy.optimize import linprog
from validator import validate_schedule


def optimize_energy(hours, battery, directives=None):
    n = len(hours)
    if directives is None:
        directives = []

    # Decision variables for each hour
    #
    # battery_action:
    #   positive = charging
    #   negative = discharging
    #
    grid = list(range(n))
    solar_used = list(range(n, 2 * n))
    battery_action = list(range(2 * n, 3 * n))
    battery_energy = list(range(3 * n, 4 * n))

    total_variables = 4 * n

    print("Number of variables:", total_variables)
    print("Grid variables:", grid)
    print("Solar variables:", solar_used)
    print("Battery action variables:", battery_action)
    print("Battery energy variables:", battery_energy)

    # ---------------------------------------------------------
    # OBJECTIVE
    # ---------------------------------------------------------
    # Minimize total grid electricity cost:
    #
    # total_cost = sum(grid[h] * tariff[h])
    #
    objective = np.zeros(total_variables)

    for h in range(n):
        objective[grid[h]] = hours[h]["tariff_bdt_per_kwh"]

    # ---------------------------------------------------------
    # EQUALITY CONSTRAINTS
    # ---------------------------------------------------------

    equality_constraints = []
    equality_values = []

    # 1. Hourly energy balance
    #
    # grid + solar_used = demand + battery_action
    #
    # Rearranged:
    # grid + solar_used - battery_action = demand

    for h in range(n):
        row = np.zeros(total_variables)

        row[grid[h]] = 1
        row[solar_used[h]] = 1
        row[battery_action[h]] = -1

        equality_constraints.append(row)
        equality_values.append(hours[h]["demand_kwh"])


    # 2. Battery energy evolution
    #
    # Hour 0:
    # battery_energy[0] = initial_energy + battery_action[0]
    #
    # Hours 1 onward:
    # battery_energy[h] = battery_energy[h-1] + battery_action[h]

    row = np.zeros(total_variables)

    row[battery_energy[0]] = 1
    row[battery_action[0]] = -1

    equality_constraints.append(row)
    equality_values.append(battery["initial_energy_kwh"])

    for h in range(1, n):
        row = np.zeros(total_variables)

        row[battery_energy[h]] = 1
        row[battery_energy[h - 1]] = -1
        row[battery_action[h]] = -1

        equality_constraints.append(row)
        equality_values.append(0)

    # 3. Final battery energy must equal initial battery energy

    row = np.zeros(total_variables)

    row[battery_energy[n - 1]] = 1

    equality_constraints.append(row)
    equality_values.append(battery["initial_energy_kwh"])

    # ---------------------------------------------------------
    # INEQUALITY CONSTRAINTS
    # ---------------------------------------------------------

    inequality_constraints = []
    inequality_values = []

    # no_charge_window
    #
    # battery_action <= 0
    #
    # Since positive = charging and negative = discharging,
    # this prevents charging during the specified hours.

    for directive in directives:
        if directive["directive_type"] == "no_charge_window":
            for h in directive["structured_adjustment"]["hours"]:
                row = np.zeros(total_variables)

                row[battery_action[h]] = 1

                inequality_constraints.append(row)
                inequality_values.append(0)


    # no_discharge_window
    #
    # -battery_action <= 0
    #
    # Equivalent to:
    # battery_action >= 0
    #
    # This prevents discharging during the specified hours.

    for directive in directives:
        if directive["directive_type"] == "no_discharge_window":
            for h in directive["structured_adjustment"]["hours"]:
                row = np.zeros(total_variables)

                row[battery_action[h]] = -1

                inequality_constraints.append(row)
                inequality_values.append(0)


    # max_grid_window
    #
    # grid <= max_grid_kwh
    #
    # This limits grid usage during the specified hours.

    for directive in directives:
        if directive["directive_type"] == "max_grid_window":
            max_grid = directive["structured_adjustment"]["max_grid_kwh"]

            for h in directive["structured_adjustment"]["hours"]:
                row = np.zeros(total_variables)

                row[grid[h]] = 1

                inequality_constraints.append(row)
                inequality_values.append(max_grid)
    # ---------------------------------------------------------
    # VARIABLE BOUNDS
    # ---------------------------------------------------------
    #
    # For now:
    # grid >= 0
    # solar_used >= 0
    # battery_action can be positive or negative
    # battery_energy >= 0
    #
    # We'll add the actual battery limits and solar limits next.
    #
    bounds = []

    for h in range(n):
        bounds.append((0, None))       # grid

    for h in range(n):
        effective_solar = hours[h]["solar_kwh"]

        for directive in directives:
            if directive["directive_type"] == "solar_reduction":
                adjustment = directive["structured_adjustment"]

                if h in adjustment["hours"]:
                    effective_solar *= adjustment["factor"]

        bounds.append((0, effective_solar))

    for h in range(n):
        bounds.append((
            -battery["max_discharge_kwh_per_hour"],
            battery["max_charge_kwh_per_hour"],
        ))  # battery_action

    for h in range(n):
        minimum_energy = battery["minimum_energy_kwh"]

        for directive in directives:
            if directive["directive_type"] == "minimum_battery_reserve":
                adjustment = directive["structured_adjustment"]

                if h in adjustment["hours"]:
                    minimum_energy = max(
                        minimum_energy,
                        adjustment["minimum_battery_kwh"],
                    )

        bounds.append((
            minimum_energy,
            battery["capacity_kwh"],
        ))

    # ---------------------------------------------------------
    # SOLVE
    # ---------------------------------------------------------

    result = linprog(
        c=objective,
        A_eq=np.array(equality_constraints),
        b_eq=np.array(equality_values),
        A_ub=np.array(inequality_constraints) if inequality_constraints else None,
        b_ub=np.array(inequality_values) if inequality_values else None,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    validate_schedule(
        hours,
        battery,
        directives,
        result,
    )

    print("Schedule validation successful!")

    print("Optimization successful!")
    print("Objective value:", result.fun)
    print("\nGrid:", result.x[grid])
    print("Solar used:", result.x[solar_used])
    print("Battery action:", result.x[battery_action])
    print("Battery energy:", result.x[battery_energy])

    return result


if __name__ == "__main__":
    # Temporary test data.
    # This is NOT the final challenge input.
    test_hours = [
        {
            "hour": h,
            "demand_kwh": 10,
            "solar_kwh": 5,
            "tariff_bdt_per_kwh": 10,
        }
        for h in range(24)
    ]

    test_battery = {
        "capacity_kwh": 20,
        "initial_energy_kwh": 10,
        "minimum_energy_kwh": 2,
        "max_charge_kwh_per_hour": 5,
        "max_discharge_kwh_per_hour": 5,
    }

    test_directives = [
        {
            "directive_type": "solar_reduction",
            "structured_adjustment": {
                "hours": [6, 7],
                "factor": 0.2,
            },
        },
        {
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {
                "hours": [8, 9],
                "minimum_battery_kwh": 8,
            },
        },
        {
            "directive_type": "no_charge_window",
            "structured_adjustment": {
                "hours": [10, 11],
            },
        },
        {
            "directive_type": "no_discharge_window",
            "structured_adjustment": {
                "hours": [12, 13],
            },
        },
        {
            "directive_type": "max_grid_window",
            "structured_adjustment": {
                "hours": [14, 15],
                "max_grid_kwh": 6,
            },
        },
    ]

    result = optimize_energy(
        test_hours,
        test_battery,
        test_directives,
    )