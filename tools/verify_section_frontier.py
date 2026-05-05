from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig
from src.core.section_calculator import SectionCalculator
from src.core.section_frontier import FrontierConfig, SectionCandidate, SectionFrontierBuilder


def dominates(a: SectionCandidate, b: SectionCandidate) -> bool:
    sa = a.section
    sb = b.section
    eps = 1e-8
    return (
        sa.weight_kg_m <= sb.weight_kg_m + eps
        and sa.EI_vertical_Nmm2 >= sb.EI_vertical_Nmm2 - eps
        and sa.EI_foreaft_Nmm2 >= sb.EI_foreaft_Nmm2 - eps
        and sa.D_over_t_conservative <= sb.D_over_t_conservative + eps
        and (
            sa.weight_kg_m < sb.weight_kg_m - eps
            or sa.EI_vertical_Nmm2 > sb.EI_vertical_Nmm2 + eps
            or sa.EI_foreaft_Nmm2 > sb.EI_foreaft_Nmm2 + eps
            or sa.D_over_t_conservative < sb.D_over_t_conservative - eps
        )
    )


def enumerate_all(grammar: LaminateGrammar, calc: SectionCalculator, diameter: float) -> list[SectionCandidate]:
    states = [grammar.initial_state()]
    all_states = list(states)
    for _ in range(grammar.config.max_total_cap_plies):
        next_states = []
        for state in states:
            next_states.extend(grammar.next_states(state))
        states = next_states
        all_states.extend(states)

    candidates = []
    for state in all_states:
        stack = grammar.build_stack(state.cap_phis)
        section = calc.evaluate(diameter, stack)
        candidates.append(SectionCandidate(state.cap_phis, stack, section, state))
    return candidates


def nondominated(candidates: list[SectionCandidate]) -> list[SectionCandidate]:
    result = []
    for candidate in candidates:
        if not any(dominates(other, candidate) for other in candidates if other is not candidate):
            result.append(candidate)
    return result


def main() -> None:
    grammar = LaminateGrammar(
        LaminateGrammarConfig(phi_min_deg=60, phi_max_deg=180, phi_step_deg=60, max_concentrated_run=2, max_total_cap_plies=3)
    )
    calc = SectionCalculator()
    builder = SectionFrontierBuilder(
        grammar=grammar,
        calculator=calc,
        config=FrontierConfig(log_ei_bin_width=1e-6, max_states_per_depth=10000, max_frontier_size=10000),
    )

    diameter = 80.0
    frontier = builder.build(diameter)
    exact = nondominated(enumerate_all(grammar, calc, diameter))

    frontier_phis = {candidate.cap_phis for candidate in frontier}
    exact_phis = {candidate.cap_phis for candidate in exact}
    missing = exact_phis - frontier_phis
    if missing:
        raise AssertionError(f"frontier missed nondominated states: {sorted(missing)[:5]}")

    for i, a in enumerate(frontier):
        for b in frontier[i + 1 :]:
            if dominates(a, b) or dominates(b, a):
                raise AssertionError("frontier contains dominated pair")

    print(f"section frontier verification complete: {len(frontier)} candidates")


if __name__ == "__main__":
    main()
