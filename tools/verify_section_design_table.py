from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.exact_section_enumerator import ExactSectionEnumerator, remove_dominated_final
from src.core.laminate_grammar import LaminateGrammarConfig
from src.core.section_design_table import SectionDesignTable
from src.core.section_table_io import load_section_design_table


def main() -> None:
    grammar_config = LaminateGrammarConfig(
        phi_min_deg=20,
        phi_max_deg=180,
        phi_step_deg=60,
        max_concentrated_run=10,
        max_total_cap_plies=2,
    )
    diameter = 80.0
    enumerator = ExactSectionEnumerator.from_configs(grammar_config=grammar_config)
    all_candidates = enumerator.enumerate(diameter)
    frontier = remove_dominated_final(all_candidates)
    table = SectionDesignTable.from_candidates(diameter, frontier)

    assert len(all_candidates) == 18
    assert len(frontier) > 0
    assert len(table.entries) > 0

    verify_query_satisfies_requirement(table)
    verify_weight_monotonicity(table)
    verify_json_roundtrip(diameter, grammar_config, all_candidates, frontier, table)

    print(
        "section design table verification complete: "
        f"all={len(all_candidates)} frontier={len(frontier)} table={len(table.entries)}"
    )


def verify_query_satisfies_requirement(table: SectionDesignTable) -> None:
    requirements = (0.0, 1.0e9, 2.0e9, 5.0e9, 1.0e10)
    for requirement in requirements:
        result = table.query(requirement)
        if not result.found:
            raise AssertionError(f"query unexpectedly failed for EI={requirement:.3e}")
        assert result.candidate is not None
        if result.candidate.EI_vertical_Nmm2 < requirement:
            raise AssertionError(
                f"query returned insufficient EI: "
                f"actual={result.candidate.EI_vertical_Nmm2:.3e}, req={requirement:.3e}"
            )


def verify_weight_monotonicity(table: SectionDesignTable) -> None:
    weights = [entry.candidate.weight_kg_m for entry in table.entries]
    for prev, cur in zip(weights[:-1], weights[1:]):
        if cur + 1e-12 < prev:
            raise AssertionError(f"table weights decreased with higher EI requirement: {prev} -> {cur}")


def verify_json_roundtrip(
    diameter: float,
    grammar_config: LaminateGrammarConfig,
    all_candidates,
    frontier,
    table: SectionDesignTable,
) -> None:
    payload = {
        "metadata": {
            "diameter_mm": diameter,
            "grammar_config": grammar_config.__dict__,
            "enumeration_config": {},
            "table_config": table.config.__dict__,
        },
        "summary": {
            "all_candidate_count": len(all_candidates),
            "frontier_candidate_count": len(frontier),
            "table_entry_count": len(table.entries),
            "elapsed_s": 0.0,
        },
        "frontier_candidates": table_records_from_candidates(frontier),
        "table_entries": table.to_records(),
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "table.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        stored = load_section_design_table(path)

    if stored.diameter_mm != diameter:
        raise AssertionError("roundtrip diameter mismatch")
    if len(stored.entries) != len(table.entries):
        raise AssertionError("roundtrip table entry count mismatch")

    requirement = 2.0e9
    original = table.query(requirement)
    loaded = stored.query(requirement)
    if not original.found or not loaded.found:
        raise AssertionError("roundtrip query failed")
    assert original.candidate is not None
    assert loaded.section is not None
    if tuple(original.candidate.cap_phis) != loaded.section.cap_phis:
        raise AssertionError("roundtrip query returned a different cap sequence")


def table_records_from_candidates(candidates) -> list[dict]:
    records = []
    for candidate in candidates:
        section = candidate.section
        records.append(
            {
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


if __name__ == "__main__":
    main()
