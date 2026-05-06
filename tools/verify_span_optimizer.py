from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig
from src.core.section_calculator import SectionCalculator
from src.core.section_frontier import FrontierConfig, SectionFrontierBuilder
from src.core.span_optimizer import SpanOptimizer, SpanOptimizerConfig, SpanStation, transition_allowed


def build_frontier():
    grammar = LaminateGrammar(
        LaminateGrammarConfig(phi_min_deg=60, phi_max_deg=180, phi_step_deg=60, max_concentrated_run=2, max_total_cap_plies=3)
    )
    builder = SectionFrontierBuilder(
        grammar=grammar,
        calculator=SectionCalculator(),
        config=FrontierConfig(log_ei_bin_width=1e-6, max_states_per_depth=10000, max_frontier_size=10000),
    )
    return tuple(builder.build(80.0))


def solve(delta_allow_m: float):
    candidates = build_frontier()
    stations = (
        SpanStation(y_m=1.0, width_m=1.0, moment_Nm=500.0, candidates=candidates),
        SpanStation(y_m=2.0, width_m=1.0, moment_Nm=200.0, candidates=candidates),
        SpanStation(y_m=3.0, width_m=1.0, moment_Nm=50.0, candidates=candidates),
    )
    opt = SpanOptimizer(
        SpanOptimizerConfig(
            delta_allow_m=delta_allow_m,
            sigma_allow_MPa=300.0,
            max_candidates_per_station=100,
            max_exhaustive_combinations=1_000_000,
        )
    )
    return opt.solve(stations)


def solve_with_transition_constraints():
    candidates = build_frontier()
    stations = (
        SpanStation(y_m=1.0, width_m=1.0, moment_Nm=500.0, candidates=candidates),
        SpanStation(y_m=2.0, width_m=1.0, moment_Nm=200.0, candidates=candidates),
        SpanStation(y_m=3.0, width_m=1.0, moment_Nm=50.0, candidates=candidates),
    )
    opt = SpanOptimizer(
        SpanOptimizerConfig(
            delta_allow_m=5.0,
            sigma_allow_MPa=300.0,
            max_candidates_per_station=100,
            max_exhaustive_combinations=1_000_000,
            enforce_monotone_cap_plies=True,
            require_common_phi_nonincreasing=True,
            max_cap_ply_drop_per_transition=1,
            max_phi_change_deg_per_transition=120,
        )
    )
    return opt.solve(stations)


def verify_transition_rules() -> None:
    by_phis = {candidate.cap_phis: candidate for candidate in build_frontier()}
    assert transition_allowed(
        by_phis[(180, 120)],
        by_phis[(120,)],
        enforce_monotone_cap_plies=True,
        require_common_phi_nonincreasing=True,
        max_cap_ply_drop_per_transition=1,
        max_phi_change_deg_per_transition=60,
    )
    assert not transition_allowed(
        by_phis[(60,)],
        by_phis[(120,)],
        require_common_phi_nonincreasing=True,
    )
    assert not transition_allowed(
        by_phis[(120,)],
        by_phis[(120, 60)],
        enforce_monotone_cap_plies=True,
    )
    assert not transition_allowed(
        by_phis[(180, 120, 60)],
        by_phis[()],
        max_cap_ply_drop_per_transition=2,
    )
    assert not transition_allowed(
        by_phis[(180,)],
        by_phis[(60,)],
        max_phi_change_deg_per_transition=60,
    )


def main() -> None:
    loose = solve(delta_allow_m=5.0)
    tight = solve(delta_allow_m=0.2)
    constrained = solve_with_transition_constraints()
    verify_transition_rules()
    assert loose.feasible
    assert tight.feasible
    assert constrained.feasible
    assert tight.total_weight_kg >= loose.total_weight_kg
    assert tight.deflection_tip_m <= loose.deflection_tip_m
    assert np.isfinite(tight.total_weight_kg)
    for rootward, tipward in zip(constrained.choices[:-1], constrained.choices[1:]):
        assert transition_allowed(
            rootward,
            tipward,
            enforce_monotone_cap_plies=True,
            require_common_phi_nonincreasing=True,
            max_cap_ply_drop_per_transition=1,
            max_phi_change_deg_per_transition=120,
        )
    print(
        "span optimizer verification complete: "
        f"loose={loose.total_weight_kg:.3f}kg "
        f"tight={tight.total_weight_kg:.3f}kg "
        f"constrained={constrained.total_weight_kg:.3f}kg"
    )


if __name__ == "__main__":
    main()
