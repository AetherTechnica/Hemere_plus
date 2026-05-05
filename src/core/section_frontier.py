from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig, LaminateState
from src.core.section_calculator import SectionCalculator
from src.core.section_types import LaminateStack, SectionResult


@dataclass(frozen=True)
class FrontierConfig:
    log_ei_bin_width: float = 0.015
    max_states_per_depth: int = 2500
    max_frontier_size: int = 5000
    max_d_over_t: float | None = None
    min_ei_vertical_Nmm2: float = 0.0
    min_ei_foreaft_Nmm2: float = 0.0
    approximate_prefix_pruning: bool = False
    approximate_bin_pruning: bool = False


@dataclass(frozen=True)
class SectionCandidate:
    cap_phis: tuple[int, ...]
    stack: LaminateStack
    section: SectionResult
    grammar_state: LaminateState

    @property
    def weight_kg_m(self) -> float:
        return self.section.weight_kg_m

    @property
    def EI_vertical_Nmm2(self) -> float:
        return self.section.EI_vertical_Nmm2

    @property
    def EI_foreaft_Nmm2(self) -> float:
        return self.section.EI_foreaft_Nmm2


class SectionFrontierBuilder:
    def __init__(
        self,
        grammar: LaminateGrammar | None = None,
        calculator: SectionCalculator | None = None,
        config: FrontierConfig | None = None,
    ):
        self.grammar = grammar or LaminateGrammar()
        self.calculator = calculator or SectionCalculator()
        self.config = config or FrontierConfig()
        self._cache: dict[tuple[float, tuple[int, ...]], SectionCandidate] = {}

    def build(self, diameter_mm: float) -> list[SectionCandidate]:
        initial = self.grammar.initial_state()
        current_level = [self._candidate_for(diameter_mm, initial)]
        all_candidates = list(current_level)

        max_depth = self.grammar.config.max_total_cap_plies
        for _ in range(max_depth):
            next_level: list[SectionCandidate] = []
            for candidate in current_level:
                for next_state in self.grammar.next_states(candidate.grammar_state):
                    next_candidate = self._candidate_for(diameter_mm, next_state)
                    if self._passes_hard_filters(next_candidate):
                        next_level.append(next_candidate)

            if not next_level:
                break

            current_level = self._prune_level(next_level)
            all_candidates.extend(current_level)

        return self._global_frontier(all_candidates)

    def _candidate_for(self, diameter_mm: float, state: LaminateState) -> SectionCandidate:
        key = (round(float(diameter_mm), 9), state.cap_phis)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        stack = self.grammar.build_stack(state.cap_phis)
        section = self.calculator.evaluate(diameter_mm, stack)
        candidate = SectionCandidate(
            cap_phis=state.cap_phis,
            stack=stack,
            section=section,
            grammar_state=state,
        )
        self._cache[key] = candidate
        return candidate

    def _passes_hard_filters(self, candidate: SectionCandidate) -> bool:
        cfg = self.config
        section = candidate.section
        if cfg.max_d_over_t is not None and section.D_over_t_conservative > cfg.max_d_over_t:
            return False
        if section.EI_vertical_Nmm2 < cfg.min_ei_vertical_Nmm2:
            return False
        if section.EI_foreaft_Nmm2 < cfg.min_ei_foreaft_Nmm2:
            return False
        return True

    def _prune_level(self, candidates: list[SectionCandidate]) -> list[SectionCandidate]:
        if not self.config.approximate_prefix_pruning:
            candidates.sort(key=lambda c: (len(c.cap_phis), c.weight_kg_m, -c.EI_vertical_Nmm2))
            return candidates[: self.config.max_states_per_depth]

        by_key: dict[tuple[int, int, int, int], SectionCandidate] = {}
        for candidate in candidates:
            key = (
                candidate.grammar_state.last_phi_in_run,
                candidate.grammar_state.concentrated_run_length,
                self._log_bin(candidate.EI_vertical_Nmm2),
                self._log_bin(candidate.EI_foreaft_Nmm2),
            )
            old = by_key.get(key)
            if old is None or candidate.weight_kg_m < old.weight_kg_m:
                by_key[key] = candidate

        kept = list(by_key.values())
        kept = self._remove_dominated(kept)
        if len(kept) > self.config.max_states_per_depth:
            kept.sort(key=lambda c: (c.weight_kg_m, -c.EI_vertical_Nmm2))
            kept = kept[: self.config.max_states_per_depth]
        return kept

    def _global_frontier(self, candidates: list[SectionCandidate]) -> list[SectionCandidate]:
        filtered = [c for c in candidates if self._passes_hard_filters(c)]
        frontier = self._remove_dominated(filtered)
        if not self.config.approximate_bin_pruning:
            frontier.sort(key=lambda c: (c.weight_kg_m, c.section.D_over_t_conservative))
            return frontier[: self.config.max_frontier_size]

        by_bin: dict[tuple[int, int], SectionCandidate] = {}
        for candidate in frontier:
            key = (self._log_bin(candidate.EI_vertical_Nmm2), self._log_bin(candidate.EI_foreaft_Nmm2))
            old = by_bin.get(key)
            if old is None or candidate.weight_kg_m < old.weight_kg_m:
                by_bin[key] = candidate
        result = list(by_bin.values())
        result.sort(key=lambda c: (c.weight_kg_m, c.section.D_over_t_conservative))
        if len(result) > self.config.max_frontier_size:
            result = result[: self.config.max_frontier_size]
        return result

    def _remove_dominated(self, candidates: list[SectionCandidate]) -> list[SectionCandidate]:
        ordered = sorted(candidates, key=lambda c: (c.weight_kg_m, -c.EI_vertical_Nmm2, -c.EI_foreaft_Nmm2))
        kept: list[SectionCandidate] = []
        for candidate in ordered:
            if any(self._dominates(other, candidate) for other in kept):
                continue
            kept = [other for other in kept if not self._dominates(candidate, other)]
            kept.append(candidate)
        return kept

    def _dominates(self, a: SectionCandidate, b: SectionCandidate) -> bool:
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

    def _log_bin(self, value: float) -> int:
        safe = max(float(value), 1e-30)
        return int(math.floor(math.log10(safe) / self.config.log_ei_bin_width))


def make_default_frontier_builder(
    grammar_config: LaminateGrammarConfig | None = None,
    frontier_config: FrontierConfig | None = None,
) -> SectionFrontierBuilder:
    return SectionFrontierBuilder(
        grammar=LaminateGrammar(grammar_config),
        calculator=SectionCalculator(),
        config=frontier_config,
    )
