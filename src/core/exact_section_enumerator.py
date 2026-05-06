from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig, LaminateState
from src.core.section_calculator import SectionCalculator
from src.core.section_frontier import SectionCandidate


@dataclass(frozen=True)
class ExactEnumerationConfig:
    max_d_over_t: float | None = None
    min_ei_vertical_Nmm2: float = 0.0
    min_ei_foreaft_Nmm2: float = 0.0


class ExactSectionEnumerator:
    """
    Enumerate completed laminate section candidates without prefix pruning.

    This is the accuracy-first path: every grammar-valid cap sequence up to
    max_total_cap_plies is evaluated as a completed stack. Only hard filters
    that cannot become better after completion are applied after evaluation.
    """

    def __init__(
        self,
        grammar: LaminateGrammar | None = None,
        calculator: SectionCalculator | None = None,
        config: ExactEnumerationConfig | None = None,
    ):
        self.grammar = grammar or LaminateGrammar()
        self.calculator = calculator or SectionCalculator()
        self.config = config or ExactEnumerationConfig()

    @classmethod
    def from_configs(
        cls,
        grammar_config: LaminateGrammarConfig | None = None,
        enumeration_config: ExactEnumerationConfig | None = None,
    ) -> "ExactSectionEnumerator":
        return cls(
            grammar=LaminateGrammar(grammar_config),
            calculator=SectionCalculator(),
            config=enumeration_config,
        )

    def enumerate(self, diameter_mm: float) -> list[SectionCandidate]:
        return list(self.iter_candidates(diameter_mm))

    def iter_candidates(self, diameter_mm: float) -> Iterator[SectionCandidate]:
        for state in self.iter_states():
            stack = self.grammar.build_stack(state.cap_phis)
            section = self.calculator.evaluate(diameter_mm, stack)
            candidate = SectionCandidate(
                cap_phis=state.cap_phis,
                stack=stack,
                section=section,
                grammar_state=state,
            )
            if self._passes_hard_filters(candidate):
                yield candidate

    def iter_states(self) -> Iterator[LaminateState]:
        current_level = [self.grammar.initial_state()]
        yield current_level[0]

        for _ in range(self.grammar.config.max_total_cap_plies):
            next_level: list[LaminateState] = []
            for state in current_level:
                next_level.extend(self.grammar.next_states(state))
            if not next_level:
                break
            for state in next_level:
                yield state
            current_level = next_level

    def _passes_hard_filters(self, candidate: SectionCandidate) -> bool:
        section = candidate.section
        cfg = self.config
        if cfg.max_d_over_t is not None and section.D_over_t_conservative > cfg.max_d_over_t:
            return False
        if section.EI_vertical_Nmm2 < cfg.min_ei_vertical_Nmm2:
            return False
        if section.EI_foreaft_Nmm2 < cfg.min_ei_foreaft_Nmm2:
            return False
        return True


def remove_dominated_final(candidates: list[SectionCandidate]) -> list[SectionCandidate]:
    ordered = sorted(candidates, key=lambda c: (c.weight_kg_m, -c.EI_vertical_Nmm2, -c.EI_foreaft_Nmm2))
    kept: list[SectionCandidate] = []
    for candidate in ordered:
        if any(dominates_final(other, candidate) for other in kept):
            continue
        kept = [other for other in kept if not dominates_final(candidate, other)]
        kept.append(candidate)
    kept.sort(key=lambda c: (c.weight_kg_m, c.section.D_over_t_conservative, -c.EI_vertical_Nmm2))
    return kept


def dominates_final(a: SectionCandidate, b: SectionCandidate) -> bool:
    eps = 1e-9
    sa = a.section
    sb = b.section
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
