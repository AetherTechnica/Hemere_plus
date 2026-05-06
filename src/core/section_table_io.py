from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StoredSectionRecord:
    cap_phis: tuple[int, ...]
    weight_kg_m: float
    EI_vertical_Nmm2: float
    EI_foreaft_Nmm2: float
    D_over_t_conservative: float
    t_min_mm: float
    t_max_mm: float
    layer_count_total: int
    cap_layer_count: int


@dataclass(frozen=True)
class StoredSectionTableEntry:
    required_ei_vertical_bin: int
    required_ei_vertical_Nmm2: float
    section: StoredSectionRecord


@dataclass(frozen=True)
class StoredSectionQueryResult:
    section: StoredSectionRecord | None
    reason: str

    @property
    def found(self) -> bool:
        return self.section is not None


class StoredSectionDesignTable:
    """
    Read-only design table loaded from build_exact_section_tables.py JSON output.
    """

    def __init__(
        self,
        metadata: dict[str, Any],
        summary: dict[str, Any],
        entries: tuple[StoredSectionTableEntry, ...],
        frontier_records: tuple[StoredSectionRecord, ...] = tuple(),
    ):
        self.metadata = metadata
        self.summary = summary
        self.entries = tuple(sorted(entries, key=lambda e: e.required_ei_vertical_bin))
        self.frontier_records = frontier_records

    @property
    def diameter_mm(self) -> float:
        return float(self.metadata["diameter_mm"])

    def query(
        self,
        EI_vertical_req_Nmm2: float,
        min_foreaft_ratio: float | None = None,
        min_ei_foreaft_Nmm2: float = 0.0,
        max_d_over_t: float | None = None,
    ) -> StoredSectionQueryResult:
        best: StoredSectionRecord | None = None
        for entry in self.entries:
            section = entry.section
            if section.EI_vertical_Nmm2 < EI_vertical_req_Nmm2:
                continue
            if section.EI_foreaft_Nmm2 < min_ei_foreaft_Nmm2:
                continue
            if min_foreaft_ratio is not None:
                if section.EI_foreaft_Nmm2 < min_foreaft_ratio * section.EI_vertical_Nmm2:
                    continue
            if max_d_over_t is not None and section.D_over_t_conservative > max_d_over_t:
                continue
            if best is None or section.weight_kg_m < best.weight_kg_m:
                best = section

        if best is None:
            return StoredSectionQueryResult(section=None, reason="no stored section satisfies query")
        return StoredSectionQueryResult(section=best, reason="ok")


def load_section_design_table(path: str | Path) -> StoredSectionDesignTable:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = tuple(parse_table_entry(item) for item in data.get("table_entries", []))
    frontier_records = tuple(parse_section_record(item) for item in data.get("frontier_candidates", []))
    return StoredSectionDesignTable(
        metadata=data["metadata"],
        summary=data["summary"],
        entries=entries,
        frontier_records=frontier_records,
    )


def load_section_design_tables_from_manifest(path: str | Path) -> dict[float, StoredSectionDesignTable]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    tables: dict[float, StoredSectionDesignTable] = {}
    for filename in manifest["files"]:
        table = load_section_design_table(manifest_path.parent / filename)
        tables[round(table.diameter_mm, 6)] = table
    return tables


def parse_table_entry(item: dict[str, Any]) -> StoredSectionTableEntry:
    return StoredSectionTableEntry(
        required_ei_vertical_bin=int(item["required_ei_vertical_bin"]),
        required_ei_vertical_Nmm2=float(item["required_ei_vertical_Nmm2"]),
        section=parse_section_record(item),
    )


def parse_section_record(item: dict[str, Any]) -> StoredSectionRecord:
    return StoredSectionRecord(
        cap_phis=tuple(int(phi) for phi in item["cap_phis"]),
        weight_kg_m=float(item["weight_kg_m"]),
        EI_vertical_Nmm2=float(item["EI_vertical_Nmm2"]),
        EI_foreaft_Nmm2=float(item["EI_foreaft_Nmm2"]),
        D_over_t_conservative=float(item["D_over_t_conservative"]),
        t_min_mm=float(item["t_min_mm"]),
        t_max_mm=float(item["t_max_mm"]),
        layer_count_total=int(item["layer_count_total"]),
        cap_layer_count=int(item["cap_layer_count"]),
    )
