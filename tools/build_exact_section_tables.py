from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.exact_section_enumerator import (
    ExactEnumerationConfig,
    ExactSectionEnumerator,
    dominates_final,
    remove_dominated_final,
)
from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig
from src.core.section_calculator import SectionCalculator
from src.core.section_types import SectionCalculatorConfig
from src.core.section_design_table import SectionDesignTable, SectionDesignTableConfig
from src.core.section_frontier import SectionCandidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build accuracy-first section design tables by complete grammar enumeration, "
            "final nondominated filtering, and post-hoc EI binning."
        )
    )
    parser.add_argument(
        "--diameters",
        default="120,109.285714,98.571429,87.857143,77.142857,66.428571,55.714286,45",
        help="Comma or space separated mandrel diameters in mm.",
    )
    parser.add_argument("--output-dir", default="results/section_tables")
    parser.add_argument("--phi-min-deg", type=int, default=20)
    parser.add_argument("--phi-step-deg", type=int, default=20)
    parser.add_argument("--max-total-cap-plies", type=int, default=8)
    parser.add_argument("--max-concentrated-run", type=int, default=10)
    parser.add_argument("--max-d-over-t", type=float, default=None)
    parser.add_argument("--min-ei-vertical", type=float, default=0.0)
    parser.add_argument("--min-ei-foreaft", type=float, default=0.0)
    parser.add_argument("--table-log-ei-bin-width", type=float, default=0.01)
    parser.add_argument("--table-max-d-over-t", type=float, default=None)
    parser.add_argument("--table-min-foreaft-ratio", type=float, default=None)
    parser.add_argument("--table-min-ei-foreaft", type=float, default=0.0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel diameter builds. Use -1 for all cores. "
            "Requires joblib; falls back to serial if joblib is unavailable."
        ),
    )
    parser.add_argument(
        "--joblib-verbose",
        type=int,
        default=10,
        help="joblib progress verbosity when --n-jobs is not 1.",
    )
    parser.add_argument(
        "--save-all-candidates",
        action="store_true",
        help="Also save all hard-filtered completed candidates. This can create large files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print per-diameter progress every N completed candidates. Use 0 to disable.",
    )
    parser.add_argument(
        "--include-debug-patches",
        action="store_true",
        help=(
            "Keep AngularPatch debug data in SectionResult. Off by default to reduce memory "
            "during long exact table builds."
        ),
    )
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
    enumeration_config = ExactEnumerationConfig(
        max_d_over_t=args.max_d_over_t,
        min_ei_vertical_Nmm2=args.min_ei_vertical,
        min_ei_foreaft_Nmm2=args.min_ei_foreaft,
    )
    table_config = SectionDesignTableConfig(
        log_ei_bin_width=args.table_log_ei_bin_width,
        max_d_over_t=args.table_max_d_over_t,
        min_foreaft_ratio=args.table_min_foreaft_ratio,
        min_ei_foreaft_Nmm2=args.table_min_ei_foreaft,
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diameters_mm": list(diameters),
        "grammar_config": grammar_config.__dict__,
        "enumeration_config": enumeration_config.__dict__,
        "table_config": table_config.__dict__,
        "files": [],
    }

    build_specs = [
        {
            "diameter_mm": diameter,
            "grammar_config": grammar_config,
            "enumeration_config": enumeration_config,
            "table_config": table_config,
            "save_all_candidates": args.save_all_candidates,
            "progress_every": args.progress_every,
            "include_debug_patches": args.include_debug_patches,
        }
        for diameter in diameters
    ]

    results = run_builds(build_specs, n_jobs=args.n_jobs, joblib_verbose=args.joblib_verbose)
    for diameter, result in zip(diameters, results):
        path = output_dir / filename_for(diameter, grammar_config, table_config)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["files"].append(str(path.name))
        print_summary_line(diameter, result, path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")


def run_builds(
    build_specs: list[dict[str, Any]],
    n_jobs: int,
    joblib_verbose: int,
) -> list[dict]:
    if n_jobs == 1:
        return [build_one_diameter(**spec) for spec in build_specs]

    try:
        from joblib import Parallel, delayed
    except ImportError:
        print("[WARN] joblib is not installed; falling back to serial execution.")
        print("[WARN] Install with: python -m pip install joblib")
        return [build_one_diameter(**spec) for spec in build_specs]

    return Parallel(n_jobs=n_jobs, verbose=joblib_verbose)(
        delayed(build_one_diameter)(**spec) for spec in build_specs
    )


def print_summary_line(diameter: float, result: dict, path: Path) -> None:
    print(
        f"D={diameter:.3f}mm "
        f"all={result['summary']['all_candidate_count']} "
        f"frontier={result['summary']['frontier_candidate_count']} "
        f"table={result['summary']['table_entry_count']} "
        f"time={result['summary']['elapsed_s']:.3f}s "
        f"-> {path}"
    )


def build_one_diameter(
    diameter_mm: float,
    grammar_config: LaminateGrammarConfig,
    enumeration_config: ExactEnumerationConfig,
    table_config: SectionDesignTableConfig,
    save_all_candidates: bool,
    progress_every: int = 10000,
    include_debug_patches: bool = False,
) -> dict:
    start = time.perf_counter()
    enumerator = ExactSectionEnumerator(
        grammar=LaminateGrammar(grammar_config),
        calculator=SectionCalculator(
            config=SectionCalculatorConfig(include_debug_patches=include_debug_patches)
        ),
        config=enumeration_config,
    )

    all_candidate_count = 0
    frontier: list[SectionCandidate] = []
    all_candidate_records: list[dict] = []

    for candidate in enumerator.iter_candidates(diameter_mm):
        all_candidate_count += 1
        frontier = update_frontier(frontier, candidate)
        if save_all_candidates:
            all_candidate_records.extend(candidates_to_records([candidate]))
        if progress_every > 0 and all_candidate_count % progress_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"D={diameter_mm:.3f}mm progress "
                f"all={all_candidate_count} frontier={len(frontier)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    frontier = remove_dominated_final(frontier)
    table = SectionDesignTable.from_candidates(
        diameter_mm=diameter_mm,
        candidates=frontier,
        config=table_config,
    )
    elapsed = time.perf_counter() - start

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "diameter_mm": diameter_mm,
            "grammar_config": grammar_config.__dict__,
            "enumeration_config": enumeration_config.__dict__,
            "table_config": table_config.__dict__,
        },
        "summary": {
            "all_candidate_count": all_candidate_count,
            "frontier_candidate_count": len(frontier),
            "table_entry_count": len(table.entries),
            "elapsed_s": elapsed,
        },
        "frontier_candidates": candidates_to_records(frontier),
        "table_entries": table.to_records(),
    }
    if save_all_candidates:
        payload["all_candidates"] = all_candidate_records
    return payload


def update_frontier(
    frontier: list[SectionCandidate],
    candidate: SectionCandidate,
) -> list[SectionCandidate]:
    if any(dominates_final(other, candidate) for other in frontier):
        return frontier
    return [other for other in frontier if not dominates_final(candidate, other)] + [candidate]


def candidates_to_records(candidates: list[SectionCandidate]) -> list[dict]:
    records = []
    for candidate in candidates:
        section = candidate.section
        records.append(
            {
                "cap_phis": list(candidate.cap_phis),
                "weight_kg_m": section.weight_kg_m,
                "EI_vertical_Nmm2": section.EI_vertical_Nmm2,
                "EI_foreaft_Nmm2": section.EI_foreaft_Nmm2,
                "EI_vertical_kgfmm2": section.EI_vertical_kgfmm2,
                "EI_foreaft_kgfmm2": section.EI_foreaft_kgfmm2,
                "D_over_t_conservative": section.D_over_t_conservative,
                "t_min_mm": section.t_min_mm,
                "t_max_mm": section.t_max_mm,
                "D_outer_min_mm": section.D_outer_min_mm,
                "D_outer_max_mm": section.D_outer_max_mm,
                "area_mm2": section.area_mm2,
                "layer_count_total": section.layer_count_total,
                "cap_layer_count": section.cap_layer_count,
            }
        )
    return records


def parse_diameters(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.replace(",", " ").split())
    if not values:
        raise ValueError("--diameters must contain at least one value")
    return values


def filename_for(
    diameter_mm: float,
    grammar_config: LaminateGrammarConfig,
    table_config: SectionDesignTableConfig,
) -> str:
    stem = (
        f"D{diameter_mm:.3f}_"
        f"phi{grammar_config.phi_step_deg}_"
        f"cap{grammar_config.max_total_cap_plies}_"
        f"bin{table_config.log_ei_bin_width:.3f}"
    ).replace(".", "p")
    return f"{stem}.json"


if __name__ == "__main__":
    main()
