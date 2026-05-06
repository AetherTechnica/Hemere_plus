from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.section_frontier import SectionCandidate


@dataclass(frozen=True)
class SectionDesignTableConfig:
    log_ei_bin_width: float = 0.01
    max_d_over_t: float | None = None
    min_foreaft_ratio: float | None = None
    min_ei_foreaft_Nmm2: float = 0.0


@dataclass(frozen=True)
class SectionDesignTableEntry:
    required_ei_vertical_bin: int
    required_ei_vertical_Nmm2: float
    candidate: SectionCandidate


@dataclass(frozen=True)
class SectionDesignQueryResult:
    candidate: SectionCandidate | None
    reason: str

    @property
    def found(self) -> bool:
        return self.candidate is not None


class SectionDesignTable:
    """
    Lookup table from required vertical EI to the lightest completed section.

    The table is built after exact enumeration/frontier generation. Binning here
    is a design-table compression step, not search-time pruning.
    """

    def __init__(
        self,
        diameter_mm: float,
        entries: tuple[SectionDesignTableEntry, ...],
        config: SectionDesignTableConfig | None = None,
    ):
        self.diameter_mm = float(diameter_mm)
        self.config = config or SectionDesignTableConfig()
        self.entries = tuple(sorted(entries, key=lambda e: e.required_ei_vertical_bin))

    @classmethod
    def from_candidates(
        cls,
        diameter_mm: float,
        candidates: list[SectionCandidate] | tuple[SectionCandidate, ...],
        config: SectionDesignTableConfig | None = None,
    ) -> "SectionDesignTable":
        cfg = config or SectionDesignTableConfig()
        filtered = [candidate for candidate in candidates if passes_table_constraints(candidate, cfg)]
        if not filtered:
            return cls(diameter_mm=diameter_mm, entries=tuple(), config=cfg)

        min_bin = log_ei_floor_bin(
            min(candidate.EI_vertical_Nmm2 for candidate in filtered),
            cfg.log_ei_bin_width,
        )
        max_bin = log_ei_ceil_bin(
            max(candidate.EI_vertical_Nmm2 for candidate in filtered),
            cfg.log_ei_bin_width,
        )

        entries: list[SectionDesignTableEntry] = []
        for bin_index in range(min_bin, max_bin + 1):
            required = 10.0 ** (bin_index * cfg.log_ei_bin_width)
            feasible = [
                candidate
                for candidate in filtered
                if candidate.EI_vertical_Nmm2 >= required
            ]
            if not feasible:
                continue
            best = min(feasible, key=lambda c: (c.weight_kg_m, c.EI_vertical_Nmm2))
            entries.append(
                SectionDesignTableEntry(
                    required_ei_vertical_bin=bin_index,
                    required_ei_vertical_Nmm2=required,
                    candidate=best,
                )
            )
        return cls(diameter_mm=diameter_mm, entries=entries, config=cfg)

    def query(
        self,
        EI_vertical_req_Nmm2: float,
        min_foreaft_ratio: float | None = None,
        min_ei_foreaft_Nmm2: float | None = None,
        max_d_over_t: float | None = None,
    ) -> SectionDesignQueryResult:
        if EI_vertical_req_Nmm2 <= 0.0:
            required_bin = -10**9
        else:
            required_bin = log_ei_ceil_bin(EI_vertical_req_Nmm2, self.config.log_ei_bin_width)

        best: SectionCandidate | None = None
        for entry in self.entries:
            candidate = entry.candidate
            if entry.required_ei_vertical_bin < required_bin:
                continue
            if candidate.EI_vertical_Nmm2 < EI_vertical_req_Nmm2:
                continue
            if not passes_query_constraints(
                candidate,
                min_foreaft_ratio=self.config.min_foreaft_ratio if min_foreaft_ratio is None else min_foreaft_ratio,
                min_ei_foreaft_Nmm2=self.config.min_ei_foreaft_Nmm2 if min_ei_foreaft_Nmm2 is None else min_ei_foreaft_Nmm2,
                max_d_over_t=self.config.max_d_over_t if max_d_over_t is None else max_d_over_t,
            ):
                continue
            if best is None or candidate.weight_kg_m < best.weight_kg_m:
                best = candidate

        if best is None:
            return SectionDesignQueryResult(candidate=None, reason="no candidate satisfies query")
        return SectionDesignQueryResult(candidate=best, reason="ok")

    def bin_floor_Nmm2(self, bin_index: int) -> float:
        return 10.0 ** (bin_index * self.config.log_ei_bin_width)

    def to_records(self) -> list[dict]:
        records = []
        for entry in self.entries:
            candidate = entry.candidate
            section = candidate.section
            records.append(
                {
                    "required_ei_vertical_bin": entry.required_ei_vertical_bin,
                    "required_ei_vertical_Nmm2": entry.required_ei_vertical_Nmm2,
                    "cap_phis": list(candidate.cap_phis),
                    "weight_kg_m": section.weight_kg_m,
                    "EI_vertical_Nmm2": section.EI_vertical_Nmm2,
                    "EI_foreaft_Nmm2": section.EI_foreaft_Nmm2,
                    "D_over_t_conservative": section.D_over_t_conservative,
                    "t_min_mm": section.t_min_mm,
                    "t_max_mm": section.t_max_mm,
                    "layer_count_total": section.layer_count_total,
                    "cap_layer_count": section.cap_layer_count,
                }
            )
        return records


def passes_table_constraints(candidate: SectionCandidate, config: SectionDesignTableConfig) -> bool:
    return passes_query_constraints(
        candidate,
        min_foreaft_ratio=config.min_foreaft_ratio,
        min_ei_foreaft_Nmm2=config.min_ei_foreaft_Nmm2,
        max_d_over_t=config.max_d_over_t,
    )


def passes_query_constraints(
    candidate: SectionCandidate,
    min_foreaft_ratio: float | None,
    min_ei_foreaft_Nmm2: float,
    max_d_over_t: float | None,
) -> bool:
    section = candidate.section
    if max_d_over_t is not None and section.D_over_t_conservative > max_d_over_t:
        return False
    if section.EI_foreaft_Nmm2 < min_ei_foreaft_Nmm2:
        return False
    if min_foreaft_ratio is not None:
        if section.EI_foreaft_Nmm2 < min_foreaft_ratio * section.EI_vertical_Nmm2:
            return False
    return True


def log_ei_floor_bin(value: float, width: float) -> int:
    if width <= 0.0:
        raise ValueError("log_ei_bin_width must be positive")
    safe = max(float(value), 1e-30)
    return int(math.floor(math.log10(safe) / width))


def log_ei_ceil_bin(value: float, width: float) -> int:
    if width <= 0.0:
        raise ValueError("log_ei_bin_width must be positive")
    safe = max(float(value), 1e-30)
    return int(math.ceil(math.log10(safe) / width))
