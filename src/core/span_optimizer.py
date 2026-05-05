from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from src.core.section_frontier import SectionCandidate


@dataclass(frozen=True)
class SpanOptimizerConfig:
    delta_allow_m: float
    sigma_allow_MPa: float | None = None
    max_d_over_t: float | None = None
    min_foreaft_ratio: float | None = None
    max_candidates_per_station: int = 80
    beam_width: int = 500
    max_exhaustive_combinations: int = 200_000


@dataclass(frozen=True)
class SpanStation:
    y_m: float
    width_m: float
    moment_Nm: float
    candidates: tuple[SectionCandidate, ...]


@dataclass(frozen=True)
class SpanOptimizationResult:
    feasible: bool
    total_weight_kg: float
    half_weight_kg: float
    deflection_tip_m: float
    choices: tuple[SectionCandidate, ...]
    reason: str


@dataclass(frozen=True)
class _PartialSpanState:
    choices: tuple[SectionCandidate, ...]
    half_weight_kg: float
    compliance_score: float


class SpanOptimizer:
    def __init__(self, config: SpanOptimizerConfig):
        if config.delta_allow_m <= 0.0:
            raise ValueError("delta_allow_m must be positive")
        self.config = config

    def solve(self, stations: tuple[SpanStation, ...] | list[SpanStation]) -> SpanOptimizationResult:
        station_tuple = tuple(stations)
        if not station_tuple:
            return SpanOptimizationResult(
                feasible=False,
                total_weight_kg=float("inf"),
                half_weight_kg=float("inf"),
                deflection_tip_m=float("inf"),
                choices=tuple(),
                reason="no stations",
            )

        filtered = [self._filter_station_candidates(station) for station in station_tuple]
        if any(len(cands) == 0 for cands in filtered):
            idx = next(i for i, cands in enumerate(filtered) if len(cands) == 0)
            return SpanOptimizationResult(
                feasible=False,
                total_weight_kg=float("inf"),
                half_weight_kg=float("inf"),
                deflection_tip_m=float("inf"),
                choices=tuple(),
                reason=f"station {idx} has no feasible section candidates",
            )

        total_combinations = 1
        for cands in filtered:
            total_combinations *= len(cands)
            if total_combinations > self.config.max_exhaustive_combinations:
                break

        if total_combinations <= self.config.max_exhaustive_combinations:
            return self._solve_exhaustive(station_tuple, filtered)
        return self._solve_beam(station_tuple, filtered)

    def _filter_station_candidates(self, station: SpanStation) -> tuple[SectionCandidate, ...]:
        kept: list[SectionCandidate] = []
        moment_Nmm = abs(station.moment_Nm) * 1000.0
        for candidate in station.candidates:
            section = candidate.section
            if self.config.max_d_over_t is not None and section.D_over_t_conservative > self.config.max_d_over_t:
                continue
            if self.config.min_foreaft_ratio is not None:
                if section.EI_foreaft_Nmm2 < self.config.min_foreaft_ratio * section.EI_vertical_Nmm2:
                    continue
            if self.config.sigma_allow_MPa is not None:
                stress = section.max_stress_vertical(moment_Nmm).max_abs_stress_MPa
                if stress > self.config.sigma_allow_MPa:
                    continue
            kept.append(candidate)

        kept.sort(key=lambda c: (c.weight_kg_m, -c.EI_vertical_Nmm2))
        return tuple(kept[: self.config.max_candidates_per_station])

    def _solve_exhaustive(
        self,
        stations: tuple[SpanStation, ...],
        candidates_by_station: list[tuple[SectionCandidate, ...]],
    ) -> SpanOptimizationResult:
        best_feasible: SpanOptimizationResult | None = None
        best_any: SpanOptimizationResult | None = None
        for choices in itertools.product(*candidates_by_station):
            result = self._evaluate_choices(stations, tuple(choices))
            if best_any is None or result.deflection_tip_m < best_any.deflection_tip_m:
                best_any = result
            if result.feasible and (
                best_feasible is None or result.total_weight_kg < best_feasible.total_weight_kg
            ):
                best_feasible = result

        if best_feasible is not None:
            return best_feasible
        assert best_any is not None
        return SpanOptimizationResult(
            feasible=False,
            total_weight_kg=best_any.total_weight_kg,
            half_weight_kg=best_any.half_weight_kg,
            deflection_tip_m=best_any.deflection_tip_m,
            choices=best_any.choices,
            reason="no combination satisfies deflection limit",
        )

    def _solve_beam(
        self,
        stations: tuple[SpanStation, ...],
        candidates_by_station: list[tuple[SectionCandidate, ...]],
    ) -> SpanOptimizationResult:
        states = [_PartialSpanState(choices=tuple(), half_weight_kg=0.0, compliance_score=0.0)]
        for station, candidates in zip(stations, candidates_by_station):
            next_states: list[_PartialSpanState] = []
            for state in states:
                for candidate in candidates:
                    weight = state.half_weight_kg + candidate.weight_kg_m * station.width_m
                    compliance = (
                        state.compliance_score
                        + abs(station.moment_Nm) / max(candidate.EI_vertical_Nmm2, 1e-30) * station.width_m
                    )
                    next_states.append(
                        _PartialSpanState(
                            choices=state.choices + (candidate,),
                            half_weight_kg=weight,
                            compliance_score=compliance,
                        )
                    )
            next_states.sort(key=lambda s: (s.half_weight_kg, s.compliance_score))
            states = next_states[: self.config.beam_width]

        evaluated = [self._evaluate_choices(stations, state.choices) for state in states]
        feasible = [result for result in evaluated if result.feasible]
        if feasible:
            return min(feasible, key=lambda r: r.total_weight_kg)
        return min(evaluated, key=lambda r: r.deflection_tip_m)

    def _evaluate_choices(
        self,
        stations: tuple[SpanStation, ...],
        choices: tuple[SectionCandidate, ...],
    ) -> SpanOptimizationResult:
        half_weight = sum(
            candidate.weight_kg_m * station.width_m
            for station, candidate in zip(stations, choices)
        )
        y = np.array([station.y_m for station in stations], dtype=float)
        moment = np.array([station.moment_Nm for station in stations], dtype=float)
        ei = np.array([candidate.EI_vertical_Nmm2 for candidate in choices], dtype=float)
        deflection = self._tip_deflection(y, moment, ei)
        feasible = deflection <= self.config.delta_allow_m
        reason = "ok" if feasible else "deflection limit exceeded"
        return SpanOptimizationResult(
            feasible=feasible,
            total_weight_kg=float(2.0 * half_weight),
            half_weight_kg=float(half_weight),
            deflection_tip_m=float(deflection),
            choices=choices,
            reason=reason,
        )

    def _tip_deflection(self, y_m: np.ndarray, moment_Nm: np.ndarray, ei_Nmm2: np.ndarray) -> float:
        order = np.argsort(y_m)
        y = y_m[order]
        moment = moment_Nm[order]
        ei_Nm2 = ei_Nmm2[order] * 1e-6
        if y[0] > 0.0:
            y = np.concatenate([[0.0], y])
            moment = np.concatenate([[moment[0]], moment])
            ei_Nm2 = np.concatenate([[ei_Nm2[0]], ei_Nm2])
        curvature = np.abs(moment) / (ei_Nm2 + 1e-30)
        theta = _cumtrapz(curvature, y)
        deflection = _cumtrapz(theta, y)
        return float(deflection[-1])


def infer_station_widths(y_m: np.ndarray) -> np.ndarray:
    y = np.array(y_m, dtype=float)
    if len(y) == 1:
        return np.array([0.0])
    bounds = np.empty(len(y) + 1)
    bounds[1:-1] = (y[:-1] + y[1:]) / 2.0
    bounds[0] = 0.0
    bounds[-1] = y[-1] + (y[-1] - bounds[-2])
    return np.diff(bounds)


def _cumtrapz(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    if len(values) == 1:
        return np.array([0.0])
    dx = np.diff(x)
    trapz = (values[:-1] + values[1:]) * dx / 2.0
    return np.concatenate([[0.0], np.cumsum(trapz)])
