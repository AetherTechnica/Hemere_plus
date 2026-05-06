from __future__ import annotations

import itertools
import math
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
    ei_diversity_bins_per_station: int = 0
    beam_width: int = 500
    max_exhaustive_combinations: int = 200_000
    enforce_monotone_cap_plies: bool = False
    require_common_phi_nonincreasing: bool = False
    max_cap_ply_drop_per_transition: int | None = None
    max_phi_change_deg_per_transition: int | None = None


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
        if config.max_cap_ply_drop_per_transition is not None and config.max_cap_ply_drop_per_transition < 0:
            raise ValueError("max_cap_ply_drop_per_transition must be non-negative")
        if config.max_phi_change_deg_per_transition is not None and config.max_phi_change_deg_per_transition < 0:
            raise ValueError("max_phi_change_deg_per_transition must be non-negative")
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

        return tuple(select_station_candidates(kept, self.config))

    def _solve_exhaustive(
        self,
        stations: tuple[SpanStation, ...],
        candidates_by_station: list[tuple[SectionCandidate, ...]],
    ) -> SpanOptimizationResult:
        best_feasible: SpanOptimizationResult | None = None
        best_any: SpanOptimizationResult | None = None
        for choices in itertools.product(*candidates_by_station):
            if not self._transitions_allowed(choices):
                continue
            result = self._evaluate_choices(stations, tuple(choices))
            if best_any is None or result.deflection_tip_m < best_any.deflection_tip_m:
                best_any = result
            if result.feasible and (
                best_feasible is None or result.total_weight_kg < best_feasible.total_weight_kg
            ):
                best_feasible = result

        if best_feasible is not None:
            return best_feasible
        if best_any is None:
            return SpanOptimizationResult(
                feasible=False,
                total_weight_kg=float("inf"),
                half_weight_kg=float("inf"),
                deflection_tip_m=float("inf"),
                choices=tuple(),
                reason="no combination satisfies transition constraints",
            )
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
                    if state.choices and not self._transition_allowed(state.choices[-1], candidate):
                        continue
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
            if not next_states:
                return SpanOptimizationResult(
                    feasible=False,
                    total_weight_kg=float("inf"),
                    half_weight_kg=float("inf"),
                    deflection_tip_m=float("inf"),
                    choices=tuple(),
                    reason="no beam state satisfies transition constraints",
                )
            states = select_beam_states(next_states, self.config.beam_width)

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

    def _transitions_allowed(self, choices: tuple[SectionCandidate, ...]) -> bool:
        return all(
            self._transition_allowed(prev, cur)
            for prev, cur in zip(choices[:-1], choices[1:])
        )

    def _transition_allowed(self, rootward: SectionCandidate, tipward: SectionCandidate) -> bool:
        return transition_allowed(
            rootward,
            tipward,
            enforce_monotone_cap_plies=self.config.enforce_monotone_cap_plies,
            require_common_phi_nonincreasing=self.config.require_common_phi_nonincreasing,
            max_cap_ply_drop_per_transition=self.config.max_cap_ply_drop_per_transition,
            max_phi_change_deg_per_transition=self.config.max_phi_change_deg_per_transition,
        )


def transition_allowed(
    rootward: SectionCandidate,
    tipward: SectionCandidate,
    *,
    enforce_monotone_cap_plies: bool = False,
    require_common_phi_nonincreasing: bool = False,
    max_cap_ply_drop_per_transition: int | None = None,
    max_phi_change_deg_per_transition: int | None = None,
) -> bool:
    """Return whether a root-to-tip section transition is manufacturable enough."""
    root_phis = rootward.cap_phis
    tip_phis = tipward.cap_phis

    if enforce_monotone_cap_plies and len(tip_phis) > len(root_phis):
        return False

    if max_cap_ply_drop_per_transition is not None:
        if len(root_phis) - len(tip_phis) > max_cap_ply_drop_per_transition:
            return False

    for root_phi, tip_phi in zip(root_phis, tip_phis):
        if require_common_phi_nonincreasing and tip_phi > root_phi:
            return False
        if max_phi_change_deg_per_transition is not None:
            if abs(tip_phi - root_phi) > max_phi_change_deg_per_transition:
                return False
    return True


def select_station_candidates(
    candidates: list[SectionCandidate],
    config: SpanOptimizerConfig,
) -> list[SectionCandidate]:
    if not candidates:
        return []
    if config.ei_diversity_bins_per_station <= 0:
        candidates.sort(key=lambda c: (c.weight_kg_m, -c.EI_vertical_Nmm2))
        return candidates[: config.max_candidates_per_station]

    by_bin: dict[int, SectionCandidate] = {}
    ei_values = [max(candidate.EI_vertical_Nmm2, 1e-30) for candidate in candidates]
    log_min = math.log10(min(ei_values))
    log_max = math.log10(max(ei_values))
    span = max(log_max - log_min, 1e-12)
    bin_count = max(config.ei_diversity_bins_per_station, 1)

    for candidate in candidates:
        normalized = (math.log10(max(candidate.EI_vertical_Nmm2, 1e-30)) - log_min) / span
        bin_index = min(bin_count - 1, max(0, int(normalized * bin_count)))
        old = by_bin.get(bin_index)
        if old is None or candidate.weight_kg_m < old.weight_kg_m:
            by_bin[bin_index] = candidate

    selected = list(by_bin.values())
    selected.sort(key=lambda c: (c.weight_kg_m, -c.EI_vertical_Nmm2))
    if len(selected) >= config.max_candidates_per_station:
        return selected[: config.max_candidates_per_station]

    selected_keys = {candidate.cap_phis for candidate in selected}
    remaining = [candidate for candidate in candidates if candidate.cap_phis not in selected_keys]
    remaining.sort(key=lambda c: (c.weight_kg_m, -c.EI_vertical_Nmm2))
    selected.extend(remaining)
    return selected[: config.max_candidates_per_station]


def select_beam_states(
    states: list[_PartialSpanState],
    beam_width: int,
) -> list[_PartialSpanState]:
    if len(states) <= beam_width:
        return states

    weight_quota = max(1, beam_width // 2)
    compliance_quota = beam_width - weight_quota
    selected: list[_PartialSpanState] = []
    selected_keys: set[tuple[tuple[int, ...], ...]] = set()

    for state in sorted(states, key=lambda s: (s.half_weight_kg, s.compliance_score))[:weight_quota]:
        key = tuple(choice.cap_phis for choice in state.choices)
        selected.append(state)
        selected_keys.add(key)

    for state in sorted(states, key=lambda s: (s.compliance_score, s.half_weight_kg)):
        if len(selected) >= beam_width:
            break
        key = tuple(choice.cap_phis for choice in state.choices)
        if key in selected_keys:
            continue
        selected.append(state)
        selected_keys.add(key)

    return selected


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
