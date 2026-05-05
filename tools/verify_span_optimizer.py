from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig
from src.core.section_calculator import SectionCalculator
from src.core.section_frontier import FrontierConfig, SectionFrontierBuilder
from src.core.span_optimizer import SpanOptimizer, SpanOptimizerConfig, SpanStation


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


def main() -> None:
    loose = solve(delta_allow_m=5.0)
    tight = solve(delta_allow_m=0.2)
    assert loose.feasible
    assert tight.feasible
    assert tight.total_weight_kg >= loose.total_weight_kg
    assert tight.deflection_tip_m <= loose.deflection_tip_m
    assert np.isfinite(tight.total_weight_kg)
    print(
        "span optimizer verification complete: "
        f"loose={loose.total_weight_kg:.3f}kg tight={tight.total_weight_kg:.3f}kg"
    )


if __name__ == "__main__":
    main()
