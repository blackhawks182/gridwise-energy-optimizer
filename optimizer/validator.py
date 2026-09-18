def validate_schedule(hours, battery, directives, result):
    """
    Validate an optimization result against the physical
    energy-system constraints and the interpreted directives.

    Raises ValueError if any constraint is violated.
    Returns True when the schedule is valid.
    """

    n = len(hours)

    # ---------------------------------------------------------
    # EXTRACT DECISION VARIABLES
    # ---------------------------------------------------------

    grid = result.x[0:n]
    solar_used = result.x[n:2 * n]
    battery_action = result.x[2 * n:3 * n]
    battery_energy = result.x[3 * n:4 * n]

    tolerance = 0.01

    # ---------------------------------------------------------
    # 1. CHECK HOURLY ENERGY BALANCE
    # ---------------------------------------------------------
    #
    # grid + solar_used - battery_action = demand
    #

    for h in range(n):
        lhs = (
            grid[h]
            + solar_used[h]
            - battery_action[h]
        )

        demand = hours[h]["demand_kwh"]

        if abs(lhs - demand) > tolerance:
            raise ValueError(
                f"Energy balance violation at hour {h}: "
                f"expected {demand}, got {lhs}"
            )

    # ---------------------------------------------------------
    # 2. CHECK SOLAR LIMIT
    # ---------------------------------------------------------

    for h in range(n):
        available_solar = hours[h]["solar_kwh"]

        effective_solar = available_solar

        for directive in directives:
            if directive["directive_type"] == "solar_reduction":
                adjustment = directive["structured_adjustment"]

                if h in adjustment["hours"]:
                    effective_solar *= adjustment["factor"]

        if solar_used[h] < -tolerance:
            raise ValueError(
                f"Negative solar usage at hour {h}: "
                f"{solar_used[h]}"
            )

        if solar_used[h] > effective_solar + tolerance:
            raise ValueError(
                f"Solar usage violation at hour {h}: "
                f"used {solar_used[h]}, "
                f"allowed {effective_solar}"
            )

    # ---------------------------------------------------------
    # 3. CHECK GRID LIMIT
    # ---------------------------------------------------------

    for h in range(n):
        if grid[h] < -tolerance:
            raise ValueError(
                f"Negative grid usage at hour {h}: "
                f"{grid[h]}"
            )

    # ---------------------------------------------------------
    # 4. CHECK BATTERY ACTION LIMITS
    # ---------------------------------------------------------

    max_charge = battery["max_charge_kwh_per_hour"]
    max_discharge = battery["max_discharge_kwh_per_hour"]

    for h in range(n):
        action = battery_action[h]

        if action > max_charge + tolerance:
            raise ValueError(
                f"Maximum charge violation at hour {h}: "
                f"charge {action}, "
                f"allowed {max_charge}"
            )

        if action < -max_discharge - tolerance:
            raise ValueError(
                f"Maximum discharge violation at hour {h}: "
                f"discharge {-action}, "
                f"allowed {max_discharge}"
            )

    # ---------------------------------------------------------
    # 5. CHECK BATTERY ENERGY LIMITS
    # ---------------------------------------------------------

    minimum_energy = battery["minimum_energy_kwh"]
    capacity = battery["capacity_kwh"]

    for h in range(n):
        energy = battery_energy[h]

        if energy < minimum_energy - tolerance:
            raise ValueError(
                f"Battery minimum-energy violation at hour {h}: "
                f"energy {energy}, "
                f"minimum {minimum_energy}"
            )

        if energy > capacity + tolerance:
            raise ValueError(
                f"Battery capacity violation at hour {h}: "
                f"energy {energy}, "
                f"capacity {capacity}"
            )

    # ---------------------------------------------------------
    # 6. CHECK BATTERY ENERGY EVOLUTION
    # ---------------------------------------------------------
    #
    # energy[h] = previous_energy + battery_action[h]
    #

    initial_energy = battery["initial_energy_kwh"]

    expected_energy = initial_energy

    for h in range(n):
        expected_energy += battery_action[h]

        if abs(battery_energy[h] - expected_energy) > tolerance:
            raise ValueError(
                f"Battery energy evolution violation at hour {h}: "
                f"expected {expected_energy}, "
                f"got {battery_energy[h]}"
            )

    # ---------------------------------------------------------
    # 7. CHECK FINAL BATTERY ENERGY
    # ---------------------------------------------------------

    if abs(
        battery_energy[n - 1] - initial_energy
    ) > tolerance:
        raise ValueError(
            f"Final battery energy violation: "
            f"expected {initial_energy}, "
            f"got {battery_energy[n - 1]}"
        )

    # ---------------------------------------------------------
    # 8. CHECK DIRECTIVE CONSTRAINTS
    # ---------------------------------------------------------

    for directive in directives:

        directive_type = directive["directive_type"]
        adjustment = directive["structured_adjustment"]

        # -----------------------------------------------------
        # no_charge_window
        # -----------------------------------------------------

        if directive_type == "no_charge_window":

            for h in adjustment["hours"]:

                if battery_action[h] > tolerance:
                    raise ValueError(
                        f"no_charge_window violation at hour {h}: "
                        f"battery action {battery_action[h]}"
                    )

        # -----------------------------------------------------
        # no_discharge_window
        # -----------------------------------------------------

        elif directive_type == "no_discharge_window":

            for h in adjustment["hours"]:

                if battery_action[h] < -tolerance:
                    raise ValueError(
                        f"no_discharge_window violation at hour {h}: "
                        f"battery action {battery_action[h]}"
                    )

        # -----------------------------------------------------
        # minimum_battery_reserve
        # -----------------------------------------------------

        elif directive_type == "minimum_battery_reserve":

            reserve = adjustment["minimum_battery_kwh"]

            for h in adjustment["hours"]:

                if battery_energy[h] < reserve - tolerance:
                    raise ValueError(
                        f"minimum_battery_reserve violation at hour {h}: "
                        f"energy {battery_energy[h]}, "
                        f"required {reserve}"
                    )

        # -----------------------------------------------------
        # max_grid_window
        # -----------------------------------------------------

        elif directive_type == "max_grid_window":

            max_grid = adjustment["max_grid_kwh"]

            for h in adjustment["hours"]:

                if grid[h] > max_grid + tolerance:
                    raise ValueError(
                        f"max_grid_window violation at hour {h}: "
                        f"grid {grid[h]}, "
                        f"maximum {max_grid}"
                    )

    # ---------------------------------------------------------
    # EVERYTHING PASSED
    # ---------------------------------------------------------

    return True