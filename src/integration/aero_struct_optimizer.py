from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.aerodynamics.aerodynamics_analyzer import AerodynamicsAnalyzer, AircraftAeroParams
from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig, LaminateState
from src.core.section_calculator import SectionCalculator
from src.core.section_frontier import FrontierConfig, SectionCandidate, SectionFrontierBuilder
from src.core.section_table_io import StoredSectionDesignTable, load_section_design_tables_from_manifest
from src.core.span_optimizer import (
    SpanOptimizationResult,
    SpanOptimizer,
    SpanOptimizerConfig,
    SpanStation,
    infer_station_widths,
)
from src.structural.structural_analyzer import StructuralAnalyzer


G0 = 9.80665


@dataclass(frozen=True)
class AeroStructDesignParams:
    W_pilot_kg: float
    W_fuselage_kg: float
    W_wing_secondary_kg: float
    span_m: float
    V_flight_ms: float
    delta_allow_m: float
    sigma_allow_MPa: float
    mandrel_diameters_mm: tuple[float, ...]
    rho_air: float = 1.154
    n_spar: int = 8
    n_aero: int = 100

    @property
    def W_fixed_kg(self) -> float:
        return self.W_pilot_kg + self.W_fuselage_kg + self.W_wing_secondary_kg


@dataclass(frozen=True)
class AeroStructOptimizerConfig:
    beta_min: float = 0.85
    beta_max: float = 1.01
    n_beta: int = 25
    max_weight_iter: int = 8
    weight_tol_kg: float = 0.02
    grammar_config: LaminateGrammarConfig = LaminateGrammarConfig(max_total_cap_plies=18)
    frontier_config: FrontierConfig = FrontierConfig(max_states_per_depth=800, max_frontier_size=1500)
    section_table_manifest_path: str | None = None
    use_table_frontier_records: bool = False
    max_candidates_per_station: int = 60
    ei_diversity_bins_per_station: int = 0
    beam_width: int = 400
    max_d_over_t: float | None = None
    min_foreaft_ratio: float | None = None
    enforce_monotone_cap_plies: bool = False
    require_common_phi_nonincreasing: bool = False
    max_cap_ply_drop_per_transition: int | None = None
    max_phi_change_deg_per_transition: int | None = None


@dataclass(frozen=True)
class AeroStructBetaResult:
    beta: float
    induced_drag_N: float
    W_spar_kg: float
    W_total_kg: float
    span_result: SpanOptimizationResult
    feasible: bool


@dataclass(frozen=True)
class AeroStructResult:
    best: AeroStructBetaResult
    sweep_results: tuple[AeroStructBetaResult, ...]


class AeroStructOptimizer:
    def __init__(self, params: AeroStructDesignParams, config: AeroStructOptimizerConfig | None = None):
        if len(params.mandrel_diameters_mm) != params.n_spar:
            raise ValueError("mandrel_diameters_mm length must match n_spar")
        self.params = params
        self.config = config or AeroStructOptimizerConfig()
        self.grammar = LaminateGrammar(self.config.grammar_config)
        self.calculator = SectionCalculator()
        self.frontier_builder = SectionFrontierBuilder(
            grammar=self.grammar,
            calculator=self.calculator,
            config=self.config.frontier_config,
        )
        self._frontier_cache: dict[float, list] = {}
        self._section_tables = self._load_section_tables()

        dummy = self._build_aero(1.0)
        self.y_aero = dummy.y
        self.struct = StructuralAnalyzer(self.y_aero)
        self.y_spar = self._spar_station_positions()
        self.spar_widths = infer_station_widths(self.y_spar)
        self.spar_idx_for_aero = self._aero_to_spar_indices()

    def optimize(self, verbose: bool = True) -> AeroStructResult:
        results: list[AeroStructBetaResult] = []
        for beta in np.linspace(self.config.beta_min, self.config.beta_max, self.config.n_beta):
            result = self._solve_beta(float(beta), verbose=verbose)
            if result is not None:
                results.append(result)

        if not results:
            raise RuntimeError("no beta produced a result")

        feasible = [result for result in results if result.feasible]
        best_pool = feasible if feasible else results
        best = min(best_pool, key=lambda r: r.induced_drag_N)
        return AeroStructResult(best=best, sweep_results=tuple(results))

    def _solve_beta(self, beta: float, verbose: bool = False) -> AeroStructBetaResult | None:
        p = self.params
        w_secondary = np.full_like(self.y_aero, p.W_wing_secondary_kg * G0 / (p.span_m / 2.0))
        w_spar_dist = np.zeros_like(self.y_aero)
        last_w_spar = 0.0
        last_span_result: SpanOptimizationResult | None = None
        last_aero_res = None

        for _ in range(self.config.max_weight_iter):
            W_spar_kg = self._weight_from_distribution(w_spar_dist)
            W_total_kg = p.W_fixed_kg + W_spar_kg
            aero = self._build_aero(W_total_kg)
            aero_res = aero.solve(beta)
            if not aero_res:
                return None
            last_aero_res = aero_res

            net_load = aero_res["lift_dist_N_m"] - w_secondary - w_spar_dist
            moment = self.struct.compute_bending_moment(net_load)
            moment_spar = np.interp(self.y_spar, self.y_aero, moment)

            span_result = self._solve_span(moment_spar)
            last_span_result = span_result
            if not span_result.choices:
                return None

            station_weights = np.array([choice.weight_kg_m for choice in span_result.choices])
            w_spar_dist = station_weights[self.spar_idx_for_aero] * G0
            new_w_spar = span_result.total_weight_kg
            if abs(new_w_spar - last_w_spar) <= self.config.weight_tol_kg:
                last_w_spar = new_w_spar
                break
            last_w_spar = new_w_spar

        if last_span_result is None or last_aero_res is None:
            return None

        W_total_final = p.W_fixed_kg + last_span_result.total_weight_kg
        aero_final = self._build_aero(W_total_final)
        aero_res_final = aero_final.solve(beta)
        if not aero_res_final:
            return None

        result = AeroStructBetaResult(
            beta=beta,
            induced_drag_N=float(aero_res_final["induced_drag_N"]),
            W_spar_kg=float(last_span_result.total_weight_kg),
            W_total_kg=float(W_total_final),
            span_result=last_span_result,
            feasible=last_span_result.feasible,
        )
        if verbose:
            ok = "OK" if result.feasible else "NG"
            print(
                f"beta={result.beta:.4f} Di={result.induced_drag_N:.4f}N "
                f"W_spar={result.W_spar_kg:.3f}kg "
                f"delta={result.span_result.deflection_tip_m*1000:.1f}mm {ok}"
            )
        return result

    def _solve_span(self, moment_spar_Nm: np.ndarray) -> SpanOptimizationResult:
        p = self.params
        stations: list[SpanStation] = []
        for y, width, moment, diameter in zip(
            self.y_spar,
            self.spar_widths,
            moment_spar_Nm,
            p.mandrel_diameters_mm,
        ):
            candidates = tuple(self._frontier_for(float(diameter)))
            stations.append(
                SpanStation(
                    y_m=float(y),
                    width_m=float(width),
                    moment_Nm=float(moment),
                    candidates=candidates,
                )
            )

        span_config = SpanOptimizerConfig(
            delta_allow_m=p.delta_allow_m,
            sigma_allow_MPa=p.sigma_allow_MPa,
            max_d_over_t=self.config.max_d_over_t,
            min_foreaft_ratio=self.config.min_foreaft_ratio,
            max_candidates_per_station=self.config.max_candidates_per_station,
            ei_diversity_bins_per_station=self.config.ei_diversity_bins_per_station,
            beam_width=self.config.beam_width,
            enforce_monotone_cap_plies=self.config.enforce_monotone_cap_plies,
            require_common_phi_nonincreasing=self.config.require_common_phi_nonincreasing,
            max_cap_ply_drop_per_transition=self.config.max_cap_ply_drop_per_transition,
            max_phi_change_deg_per_transition=self.config.max_phi_change_deg_per_transition,
        )
        return SpanOptimizer(span_config).solve(tuple(stations))

    def _frontier_for(self, diameter_mm: float):
        key = round(diameter_mm, 6)
        if key not in self._frontier_cache:
            table = self._section_tables.get(key) if self._section_tables else None
            if table is not None:
                self._frontier_cache[key] = self._candidates_from_table(table)
            else:
                self._frontier_cache[key] = self.frontier_builder.build(diameter_mm)
        return self._frontier_cache[key]

    def _load_section_tables(self) -> dict[float, StoredSectionDesignTable]:
        if not self.config.section_table_manifest_path:
            return {}
        return load_section_design_tables_from_manifest(Path(self.config.section_table_manifest_path))

    def _candidates_from_table(self, table: StoredSectionDesignTable) -> list[SectionCandidate]:
        records = table.frontier_records if self.config.use_table_frontier_records else tuple(
            entry.section for entry in table.entries
        )
        unique_cap_phis = sorted({record.cap_phis for record in records}, key=lambda phis: (len(phis), phis))
        candidates: list[SectionCandidate] = []
        for cap_phis in unique_cap_phis:
            stack = self.grammar.build_stack(cap_phis)
            section = self.calculator.evaluate(table.diameter_mm, stack)
            state = self._state_for_cap_phis(cap_phis)
            candidates.append(
                SectionCandidate(
                    cap_phis=cap_phis,
                    stack=stack,
                    section=section,
                    grammar_state=state,
                )
            )
        return candidates

    def _state_for_cap_phis(self, cap_phis: tuple[int, ...]) -> LaminateState:
        state = self.grammar.initial_state()
        for phi in cap_phis:
            next_state = self.grammar.append_phi(state, phi)
            if next_state is None:
                raise ValueError(f"stored table contains invalid cap Phi sequence: {cap_phis}")
            state = next_state
        return state

    def _build_aero(self, W_total_kg: float) -> AerodynamicsAnalyzer:
        p = self.params
        return AerodynamicsAnalyzer(
            AircraftAeroParams(
                lift_target_N=W_total_kg * G0,
                span_m=p.span_m,
                v_flight_ms=p.V_flight_ms,
                rho_air=p.rho_air,
                n_segments=p.n_aero,
            )
        )

    def _spar_station_positions(self) -> np.ndarray:
        le = self.params.span_m / 2.0
        half_width = le / self.params.n_spar / 2.0
        return np.linspace(half_width, le - half_width, self.params.n_spar)

    def _aero_to_spar_indices(self) -> np.ndarray:
        le = self.params.span_m / 2.0
        bounds = np.concatenate(
            [[0.0], (self.y_spar[:-1] + self.y_spar[1:]) / 2.0, [le + 1e-9]]
        )
        return np.searchsorted(bounds[1:], self.y_aero).clip(0, self.params.n_spar - 1)

    def _weight_from_distribution(self, w_spar_dist_N_m: np.ndarray) -> float:
        if len(w_spar_dist_N_m) < 2:
            return 0.0
        dy = self.y_aero[1] - self.y_aero[0]
        return float(np.sum(w_spar_dist_N_m / G0) * dy * 2.0)
