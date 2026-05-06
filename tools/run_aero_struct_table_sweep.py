from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.laminate_grammar import LaminateGrammarConfig
from src.core.section_frontier import FrontierConfig
from src.core.section_table_io import load_section_design_tables_from_manifest
from src.integration.aero_struct_optimizer import (
    AeroStructBetaResult,
    AeroStructDesignParams,
    AeroStructOptimizer,
    AeroStructOptimizerConfig,
    AeroStructResult,
)


ATLAS_CONDITIONS: dict[int, dict[str, Any]] = {
    1: {
        "name": "condition1_32p5m_7p2ms",
        "label": "Condition 1 (32.5m, 7.2m/s)",
        "W_pilot_kg": 58.0,
        "W_fuselage_kg": 19.0,
        "W_wing_secondary_kg": 9.0,
        "span_m": 32.5,
        "V_flight_ms": 7.2,
        "delta_allow_m": 2.2,
        "sigma_allow_MPa": 300.0,
        "mandrel_diameters_mm": tuple(float(x) for x in np.linspace(120.0, 45.0, 8)),
        "beta_ref": 0.9794,
        "Di_ref_N": 10.301,
    },
    2: {
        "name": "condition2_29m_8p3ms",
        "label": "Condition 2 (29m, 8.3m/s)",
        "W_pilot_kg": 58.0,
        "W_fuselage_kg": 19.0,
        "W_wing_secondary_kg": 9.0,
        "span_m": 29.0,
        "V_flight_ms": 8.3,
        "delta_allow_m": 2.0,
        "sigma_allow_MPa": 300.0,
        "mandrel_diameters_mm": tuple(float(x) for x in np.linspace(110.0, 40.0, 8)),
        "beta_ref": 0.9809,
        "Di_ref_N": 7.670,
    },
    3: {
        "name": "condition3_22m_11ms",
        "label": "Condition 3 (22m, 11m/s)",
        "W_pilot_kg": 55.0,
        "W_fuselage_kg": 20.0,
        "W_wing_secondary_kg": 8.0,
        "span_m": 22.0,
        "V_flight_ms": 11.0,
        "delta_allow_m": 0.9,
        "sigma_allow_MPa": 300.0,
        "mandrel_diameters_mm": tuple(float(x) for x in np.linspace(90.0, 40.0, 8)),
        "beta_ref": 0.9888,
        "Di_ref_N": 6.229,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run table-backed aero-structural beta sweeps for the opt-ATLAS reference conditions."
    )
    parser.add_argument("--table-manifest", required=True)
    parser.add_argument("--output-dir", default="results/integration_sweeps/table")
    parser.add_argument("--conditions", default="1,2,3", help="Comma or space separated condition ids.")
    parser.add_argument("--beta-min", type=float, default=0.97)
    parser.add_argument("--beta-max", type=float, default=1.01)
    parser.add_argument("--n-beta", type=int, default=9)
    parser.add_argument("--n-aero", type=int, default=100)
    parser.add_argument("--max-weight-iter", type=int, default=8)
    parser.add_argument("--weight-tol-kg", type=float, default=0.02)
    parser.add_argument("--max-candidates-per-station", type=int, default=80)
    parser.add_argument("--ei-diversity-bins-per-station", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=500)
    parser.add_argument("--max-d-over-t", type=float, default=None)
    parser.add_argument("--min-foreaft-ratio", type=float, default=None)
    parser.add_argument("--use-table-frontier-records", action="store_true")
    parser.add_argument("--allow-frontier-fallback", action="store_true")
    parser.add_argument("--enforce-monotone-cap-plies", action="store_true")
    parser.add_argument("--require-common-phi-nonincreasing", action="store_true")
    parser.add_argument("--max-cap-ply-drop-per-transition", type=int, default=None)
    parser.add_argument("--max-phi-change-deg-per-transition", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_ids = parse_condition_ids(args.conditions)
    manifest_path = Path(args.table_manifest)
    grammar_config = grammar_config_from_manifest(manifest_path)
    table_diameters = set(load_section_design_tables_from_manifest(manifest_path).keys())

    reports = []
    for condition_id in condition_ids:
        condition = ATLAS_CONDITIONS[condition_id]
        ensure_tables_cover_condition(condition, table_diameters, args.allow_frontier_fallback)
        print(f"running {condition['label']} with table manifest {manifest_path}")
        start = time.perf_counter()
        optimizer = make_optimizer(args, condition, grammar_config, manifest_path)
        result = optimizer.optimize(verbose=args.verbose)
        elapsed = time.perf_counter() - start
        report = condition_report(condition_id, condition, result, elapsed)
        reports.append(report)
        write_condition_outputs(output_dir, report)
        print_summary_line(report)

    write_summary_outputs(output_dir, args, manifest_path, reports)
    print(f"outputs -> {output_dir}")


def parse_condition_ids(raw: str) -> tuple[int, ...]:
    ids = tuple(int(part) for part in raw.replace(",", " ").split())
    if not ids:
        raise ValueError("--conditions must contain at least one condition id")
    unknown = [item for item in ids if item not in ATLAS_CONDITIONS]
    if unknown:
        raise ValueError(f"unknown condition ids: {unknown}")
    return ids


def grammar_config_from_manifest(path: Path) -> LaminateGrammarConfig:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    return LaminateGrammarConfig(**manifest["grammar_config"])


def ensure_tables_cover_condition(
    condition: dict[str, Any],
    table_diameters: set[float],
    allow_frontier_fallback: bool,
) -> None:
    missing = [
        diameter
        for diameter in condition["mandrel_diameters_mm"]
        if round(float(diameter), 6) not in table_diameters
    ]
    if missing and not allow_frontier_fallback:
        formatted = ", ".join(f"{value:.6g}" for value in missing)
        raise ValueError(
            f"table manifest does not contain all diameters for {condition['label']}: {formatted}. "
            "Build a table for this condition or pass --allow-frontier-fallback."
        )


def make_optimizer(
    args: argparse.Namespace,
    condition: dict[str, Any],
    grammar_config: LaminateGrammarConfig,
    manifest_path: Path,
) -> AeroStructOptimizer:
    diameters = tuple(float(x) for x in condition["mandrel_diameters_mm"])
    params = AeroStructDesignParams(
        W_pilot_kg=condition["W_pilot_kg"],
        W_fuselage_kg=condition["W_fuselage_kg"],
        W_wing_secondary_kg=condition["W_wing_secondary_kg"],
        span_m=condition["span_m"],
        V_flight_ms=condition["V_flight_ms"],
        delta_allow_m=condition["delta_allow_m"],
        sigma_allow_MPa=condition["sigma_allow_MPa"],
        mandrel_diameters_mm=diameters,
        n_spar=len(diameters),
        n_aero=args.n_aero,
    )
    config = AeroStructOptimizerConfig(
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        n_beta=args.n_beta,
        max_weight_iter=args.max_weight_iter,
        weight_tol_kg=args.weight_tol_kg,
        grammar_config=grammar_config,
        frontier_config=FrontierConfig(),
        section_table_manifest_path=str(manifest_path),
        use_table_frontier_records=args.use_table_frontier_records,
        max_candidates_per_station=args.max_candidates_per_station,
        ei_diversity_bins_per_station=args.ei_diversity_bins_per_station,
        beam_width=args.beam_width,
        max_d_over_t=args.max_d_over_t,
        min_foreaft_ratio=args.min_foreaft_ratio,
        enforce_monotone_cap_plies=args.enforce_monotone_cap_plies,
        require_common_phi_nonincreasing=args.require_common_phi_nonincreasing,
        max_cap_ply_drop_per_transition=args.max_cap_ply_drop_per_transition,
        max_phi_change_deg_per_transition=args.max_phi_change_deg_per_transition,
    )
    return AeroStructOptimizer(params, config)


def condition_report(
    condition_id: int,
    condition: dict[str, Any],
    result: AeroStructResult,
    elapsed_s: float,
) -> dict[str, Any]:
    best = beta_result_record(result.best)
    return {
        "condition_id": condition_id,
        "condition_name": condition["name"],
        "condition_label": condition["label"],
        "reference": {
            "beta": condition["beta_ref"],
            "Di_N": condition["Di_ref_N"],
        },
        "inputs": {
            key: value
            for key, value in condition.items()
            if key not in {"name", "label", "beta_ref", "Di_ref_N"}
        },
        "elapsed_s": elapsed_s,
        "best": best,
        "Di_error_vs_ref_percent": (
            (best["induced_drag_N"] - condition["Di_ref_N"]) / condition["Di_ref_N"] * 100.0
        ),
        "sweep_results": [beta_result_record(item) for item in result.sweep_results],
    }


def beta_result_record(item: AeroStructBetaResult) -> dict[str, Any]:
    span = item.span_result
    return {
        "beta": item.beta,
        "induced_drag_N": item.induced_drag_N,
        "W_spar_kg": item.W_spar_kg,
        "W_total_kg": item.W_total_kg,
        "feasible": item.feasible,
        "deflection_tip_m": span.deflection_tip_m,
        "span_reason": span.reason,
        "selected_cap_phis": [list(choice.cap_phis) for choice in span.choices],
        "selected_sections": [
            {
                "cap_phis": list(choice.cap_phis),
                "weight_kg_m": choice.section.weight_kg_m,
                "EI_vertical_Nmm2": choice.section.EI_vertical_Nmm2,
                "EI_foreaft_Nmm2": choice.section.EI_foreaft_Nmm2,
                "D_over_t": choice.section.D_over_t_conservative,
                "t_min_mm": choice.section.t_min_mm,
                "t_max_mm": choice.section.t_max_mm,
            }
            for choice in span.choices
        ],
    }


def write_condition_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    stem = report["condition_name"]
    (output_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / f"{stem}_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "beta",
                "induced_drag_N",
                "W_spar_kg",
                "W_total_kg",
                "deflection_tip_m",
                "feasible",
                "span_reason",
                "selected_cap_phis",
            ],
        )
        writer.writeheader()
        for row in report["sweep_results"]:
            writer.writerow(
                {
                    "beta": row["beta"],
                    "induced_drag_N": row["induced_drag_N"],
                    "W_spar_kg": row["W_spar_kg"],
                    "W_total_kg": row["W_total_kg"],
                    "deflection_tip_m": row["deflection_tip_m"],
                    "feasible": row["feasible"],
                    "span_reason": row["span_reason"],
                    "selected_cap_phis": " | ".join(format_cap_phis(seq) for seq in row["selected_cap_phis"]),
                }
            )


def write_summary_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    reports: list[dict[str, Any]],
) -> None:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "table_manifest": str(manifest_path),
        "args": serializable_args(args),
        "conditions": [
            {
                "condition_id": report["condition_id"],
                "condition_label": report["condition_label"],
                "elapsed_s": report["elapsed_s"],
                "beta_best": report["best"]["beta"],
                "Di_best_N": report["best"]["induced_drag_N"],
                "Di_ref_N": report["reference"]["Di_N"],
                "Di_error_vs_ref_percent": report["Di_error_vs_ref_percent"],
                "W_spar_kg": report["best"]["W_spar_kg"],
                "W_total_kg": report["best"]["W_total_kg"],
                "deflection_tip_m": report["best"]["deflection_tip_m"],
                "feasible": report["best"]["feasible"],
                "selected_cap_phis": report["best"]["selected_cap_phis"],
            }
            for report in reports
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition_id",
                "condition_label",
                "elapsed_s",
                "beta_best",
                "Di_best_N",
                "Di_ref_N",
                "Di_error_vs_ref_percent",
                "W_spar_kg",
                "W_total_kg",
                "deflection_tip_m",
                "feasible",
                "selected_cap_phis",
            ],
        )
        writer.writeheader()
        for row in summary["conditions"]:
            writer.writerow(
                {
                    **row,
                    "selected_cap_phis": " | ".join(format_cap_phis(seq) for seq in row["selected_cap_phis"]),
                }
            )


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def print_summary_line(report: dict[str, Any]) -> None:
    best = report["best"]
    print(
        f"{report['condition_label']}: "
        f"beta={best['beta']:.4f} "
        f"Di={best['induced_drag_N']:.4f}N "
        f"W_spar={best['W_spar_kg']:.3f}kg "
        f"delta={best['deflection_tip_m']*1000.0:.1f}mm "
        f"err={report['Di_error_vs_ref_percent']:+.2f}% "
        f"feasible={best['feasible']}"
    )


def format_cap_phis(sequence: list[int]) -> str:
    if not sequence:
        return "base"
    return "/".join(f"Phi{phi}" for phi in sequence)


if __name__ == "__main__":
    main()
