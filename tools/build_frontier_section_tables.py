from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig
from src.core.section_calculator import SectionCalculator
from src.core.section_design_table import SectionDesignTable, SectionDesignTableConfig
from src.core.section_frontier import FrontierConfig, SectionFrontierBuilder
from tools.build_exact_section_tables import candidates_to_records, filename_for, parse_diameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build practical section design tables from SectionFrontierBuilder. "
            "This is faster than exact enumeration and intended for integration sweeps."
        )
    )
    parser.add_argument("--diameters", required=True, help="Comma or space separated mandrel diameters in mm.")
    parser.add_argument("--output-dir", default="results/section_tables_frontier")
    parser.add_argument("--phi-min-deg", type=int, default=20)
    parser.add_argument("--phi-step-deg", type=int, default=20)
    parser.add_argument("--max-total-cap-plies", type=int, default=8)
    parser.add_argument("--max-concentrated-run", type=int, default=10)
    parser.add_argument("--max-states-per-depth", type=int, default=2500)
    parser.add_argument("--max-frontier-size", type=int, default=5000)
    parser.add_argument("--frontier-log-ei-bin-width", type=float, default=0.015)
    parser.add_argument("--max-d-over-t", type=float, default=None)
    parser.add_argument("--min-ei-vertical", type=float, default=0.0)
    parser.add_argument("--min-ei-foreaft", type=float, default=0.0)
    parser.add_argument("--approx-prefix-pruning", action="store_true")
    parser.add_argument("--approx-bin-pruning", action="store_true")
    parser.add_argument("--table-log-ei-bin-width", type=float, default=0.01)
    parser.add_argument("--table-max-d-over-t", type=float, default=None)
    parser.add_argument("--table-min-foreaft-ratio", type=float, default=None)
    parser.add_argument("--table-min-ei-foreaft", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diameters = parse_diameters(args.diameters)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grammar_config = LaminateGrammarConfig(
        phi_min_deg=args.phi_min_deg,
        phi_max_deg=180,
        phi_step_deg=args.phi_step_deg,
        max_concentrated_run=args.max_concentrated_run,
        max_total_cap_plies=args.max_total_cap_plies,
    )
    frontier_config = FrontierConfig(
        log_ei_bin_width=args.frontier_log_ei_bin_width,
        max_states_per_depth=args.max_states_per_depth,
        max_frontier_size=args.max_frontier_size,
        max_d_over_t=args.max_d_over_t,
        min_ei_vertical_Nmm2=args.min_ei_vertical,
        min_ei_foreaft_Nmm2=args.min_ei_foreaft,
        approximate_prefix_pruning=args.approx_prefix_pruning,
        approximate_bin_pruning=args.approx_bin_pruning,
    )
    table_config = SectionDesignTableConfig(
        log_ei_bin_width=args.table_log_ei_bin_width,
        max_d_over_t=args.table_max_d_over_t,
        min_foreaft_ratio=args.table_min_foreaft_ratio,
        min_ei_foreaft_Nmm2=args.table_min_ei_foreaft,
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "builder": "frontier",
        "diameters_mm": list(diameters),
        "grammar_config": grammar_config.__dict__,
        "frontier_config": frontier_config.__dict__,
        "table_config": table_config.__dict__,
        "files": [],
    }

    for diameter in diameters:
        payload = build_one_diameter(diameter, grammar_config, frontier_config, table_config)
        path = output_dir / filename_for(diameter, grammar_config, table_config)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["files"].append(path.name)
        summary = payload["summary"]
        print(
            f"D={diameter:.3f}mm "
            f"frontier={summary['frontier_candidate_count']} "
            f"table={summary['table_entry_count']} "
            f"time={summary['elapsed_s']:.3f}s -> {path}"
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")


def build_one_diameter(
    diameter_mm: float,
    grammar_config: LaminateGrammarConfig,
    frontier_config: FrontierConfig,
    table_config: SectionDesignTableConfig,
) -> dict:
    start = time.perf_counter()
    builder = SectionFrontierBuilder(
        grammar=LaminateGrammar(grammar_config),
        calculator=SectionCalculator(),
        config=frontier_config,
    )
    frontier = builder.build(diameter_mm)
    table = SectionDesignTable.from_candidates(diameter_mm, frontier, config=table_config)
    elapsed = time.perf_counter() - start
    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "diameter_mm": diameter_mm,
            "builder": "frontier",
            "grammar_config": grammar_config.__dict__,
            "frontier_config": frontier_config.__dict__,
            "table_config": table_config.__dict__,
        },
        "summary": {
            "frontier_candidate_count": len(frontier),
            "table_entry_count": len(table.entries),
            "elapsed_s": elapsed,
        },
        "frontier_candidates": candidates_to_records(frontier),
        "table_entries": table.to_records(),
    }


if __name__ == "__main__":
    main()
