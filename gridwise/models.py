from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator


DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


class HourInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int = Field(ge=0, le=23)
    demand_kwh: FiniteFloat = Field(ge=0)
    solar_kwh: FiniteFloat = Field(ge=0)
    tariff_bdt_per_kwh: FiniteFloat = Field(ge=0)


class BatteryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capacity_kwh: FiniteFloat = Field(gt=0)
    initial_energy_kwh: FiniteFloat = Field(ge=0)
    minimum_energy_kwh: FiniteFloat = Field(ge=0)
    max_charge_kwh_per_hour: FiniteFloat = Field(ge=0)
    max_discharge_kwh_per_hour: FiniteFloat = Field(ge=0)

    @model_validator(mode="after")
    def validate_relationships(self) -> "BatteryInput":
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh cannot exceed capacity_kwh")
        if self.initial_energy_kwh < self.minimum_energy_kwh:
            raise ValueError("initial_energy_kwh cannot be below minimum_energy_kwh")
        return self


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    scenario_id: str = Field(min_length=1)
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourInput]
    battery: BatteryInput

    @field_validator("operator_notes")
    @classmethod
    def validate_operator_notes(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for note in value:
            if not isinstance(note, str) or not note.strip():
                raise ValueError("operator_notes must contain non-empty strings")
            cleaned.append(note.strip())
        return cleaned

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, value: list[HourInput]) -> list[HourInput]:
        if len(value) != 24:
            raise ValueError("hours must contain exactly 24 entries")
        hour_numbers = [item.hour for item in value]
        if hour_numbers != list(range(24)):
            raise ValueError("hours must contain exactly 0 through 23 in ascending order")
        return value


class SolarReductionAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hours: list[int]
    factor: FiniteFloat


class MinimumBatteryReserveAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hours: list[int]
    minimum_energy_kwh: FiniteFloat


class WindowAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hours: list[int]


class MaxGridWindowAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hours: list[int]
    max_grid_kwh: FiniteFloat


Adjustment = (
    SolarReductionAdjustment
    | MinimumBatteryReserveAdjustment
    | WindowAdjustment
    | MaxGridWindowAdjustment
)


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    note_index: int = Field(ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: dict[str, Any] | None
    explanation: str = Field(min_length=1, max_length=1000)


class LLMInterpretationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directive_interpretation: list[DirectiveInterpretation]


class HourlyPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int = Field(ge=0, le=23)
    grid_kwh: FiniteFloat = Field(ge=0)
    solar_used_kwh: FiniteFloat = Field(ge=0)
    battery_action: BatteryAction
    battery_kwh: FiniteFloat = Field(ge=0)
    battery_energy_after_kwh: FiniteFloat = Field(ge=0)


class OptimizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: FiniteFloat = Field(ge=0)
    total_cost_bdt: FiniteFloat = Field(ge=0)
    peak_grid_kwh: FiniteFloat = Field(ge=0)
    plan_summary: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
