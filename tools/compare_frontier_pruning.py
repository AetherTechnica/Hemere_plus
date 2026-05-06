from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig
from src.core.section_calculator import SectionCalculator
from src.core.section_frontier import FrontierConfig, SectionCandidate, SectionFrontierBuilder
from src.core.span_optimizer import SpanOptimizationResult, SpanOptimizer, SpanOptimizerConfig, SpanStation, infer_station_widths


@dataclass(frozen=True)
class VariantConfig:
    name: str
    approximate_prefix_pruning: bool
    approximate_bin_pruning: bool


@dataclass(frozen=True)
class DiameterComparison:
    diameter_mm: float
    baseline_size: int
    variant_size: int
    baseline_time_s: float
    variant_time_s: float
    missing_from_baseline: int
    added_vs_baseline: int
    retained_baseline_fraction: float
    baseline_weight_range: tuple[float, float] | None
    variant_weight_range: tuple[float, float] | None
    baseline_ei_vertical_range: tuple[float, float] | None
    variant_ei_vertical_range: tuple[float, float] | None


@dataclass(frozen=True)
class VariantResult:
    config: VariantConfig
    frontiers_by_diameter: dict[float, list[SectionCandidate]]
    build_times_by_diameter: dict[float, float]
    span_result: SpanOptimizationResult


VARIANTS = (
    VariantConfig("baseline", False, False),
    VariantConfig("prefix", True, False),
    VariantConfig("bin", False, True),
    VariantConfig("prefix_bin", True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact-like baseline frontier generation with approximate pruning variants, "
            "including a fixed span optimizer check."
        )
    )
    parser.add_argument(
        "--diameters",
        default="120,100,80,60",
        help="Comma or space separated mandrel diameters in mm.",
    )
    parser.add_argument("--half-span-m", type=float, default=8.0)
    parser.add_argument("--root-moment-Nm", type=float, default=900.0)
    parser.add_argument("--moment-power", type=float, default=1.7)
    parser.add_argument("--delta-allow-m", type=float, default=0.8)
    parser.add_argument("--sigma-allow-MPa", type=float, default=300.0)
    parser.add_argument("--phi-min-deg", type=int, default=20)
    parser.add_argument("--phi-step-deg", type=int, default=20)
    parser.add_argument("--max-total-cap-plies", type=int, default=5)
    parser.add_argument("--max-concentrated-run", type=int, default=10)
    parser.add_argument("--max-states-per-depth", type=int, default=400)
    parser.add_argument("--max-frontier-size", type=int, default=800)
    parser.add_argument("--max-candidates-per-station", type=int, default=80)
    parser.add_argument("--ei-diversity-bins-per-station", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=500)
    parser.add_argument("--max-d-over-t", type=float, default=None)
    parser.add_argument("--min-foreaft-ratio", type=float, default=None)
    parser.add_argument("--enforce-monotone-cap-plies", action="store_true")
    parser.add_argument("--require-common-phi-nonincreasing", action="store_true")
    parser.add_argument("--max-cap-ply-drop-per-transition", type=int, default=None)
    parser.add_argument("--max-phi-change-deg-per-transition", type=int, default=None)
    parser.add_argument("--log-ei-bin-width", type=float, default=0.015)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for a machine-readable comparison report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diameters = parse_diameters(args.diameters)
    grammar_config = LaminateGrammarConfig(
        phi_min_deg=args.phi_min_deg,
        phi_max_deg=180,
        phi_step_deg=args.phi_step_deg,
        max_concentrated_run=args.max_concentrated_run,
        max_total_cap_plies=args.max_total_cap_plies,
    )

    print_config(args, diameters, grammar_config)

    results = [
        build_variant_result(args, diameters, grammar_config, variant)
        for variant in VARIANTS
    ]

    baseline = results[0]
    print_variant_summary(results)
    print_frontier_comparisons(baseline, results[1:], diameters)
    print_span_comparison(baseline, results)
    if args.output_json:
        write_json_report(args, diameters, grammar_config, results, Path(args.output_json))


def build_variant_result(
    args: argparse.Namespace,
    diameters: tuple[float, ...],
    grammar_config: LaminateGrammarConfig,
    variant: VariantConfig,
) -> VariantResult:
    frontier_config = FrontierConfig(
        log_ei_bin_width=args.log_ei_bin_width,
        max_states_per_depth=args.max_states_per_depth,
        max_frontier_size=args.max_frontier_size,
        max_d_over_t=args.max_d_over_t,
        approximate_prefix_pruning=variant.approximate_prefix_pruning,
        approximate_bin_pruning=variant.approximate_bin_pruning,
    )
    builder = SectionFrontierBuilder(
        grammar=LaminateGrammar(grammar_config),
        calculator=SectionCalculator(),
        config=frontier_config,
    )

    frontiers: dict[float, list[SectionCandidate]] = {}
    times: dict[float, float] = {}
    for diameter in diameters:
        start = time.perf_counter()
        frontier = builder.build(diameter)
        elapsed = time.perf_counter() - start
        key = round(float(diameter), 6)
        frontiers[key] = frontier
        times[key] = elapsed

    span_result = solve_span(args, diameters, frontiers)
    return VariantResult(
        config=variant,
        frontiers_by_diameter=frontiers,
        build_times_by_diameter=times,
        span_result=span_result,
    )


def solve_span(
    args: argparse.Namespace,
    diameters: tuple[float, ...],
    frontiers: dict[float, list[SectionCandidate]],
) -> SpanOptimizationResult:
    y = np.linspace(
        args.half_span_m / (2.0 * len(diameters)),
        args.half_span_m - args.half_span_m / (2.0 * len(diameters)),
        len(diameters),
    )
    widths = infer_station_widths(y)
    normalized = 1.0 - y / args.half_span_m
    moments = args.root_moment_Nm * np.maximum(normalized, 0.0) ** args.moment_power

    stations = []
    for y_m, width_m, moment_Nm, diameter in zip(y, widths, moments, diameters):
        stations.append(
            SpanStation(
                y_m=float(y_m),
                width_m=float(width_m),
                moment_Nm=float(moment_Nm),
                candidates=tuple(frontiers[round(float(diameter), 6)]),
            )
        )

    config = SpanOptimizerConfig(
        delta_allow_m=args.delta_allow_m,
        sigma_allow_MPa=args.sigma_allow_MPa,
        max_d_over_t=args.max_d_over_t,
        min_foreaft_ratio=args.min_foreaft_ratio,
        max_candidates_per_station=args.max_candidates_per_station,
        ei_diversity_bins_per_station=args.ei_diversity_bins_per_station,
        beam_width=args.beam_width,
        enforce_monotone_cap_plies=args.enforce_monotone_cap_plies,
        require_common_phi_nonincreasing=args.require_common_phi_nonincreasing,
        max_cap_ply_drop_per_transition=args.max_cap_ply_drop_per_transition,
        max_phi_change_deg_per_transition=args.max_phi_change_deg_per_transition,
    )
    return SpanOptimizer(config).solve(tuple(stations))


def print_config(
    args: argparse.Namespace,
    diameters: tuple[float, ...],
    grammar_config: LaminateGrammarConfig,
) -> None:
    print("frontier pruning comparison")
    print(f"diameters_mm={format_diameters(diameters)}")
    print(
        "grammar="
        f"phi_min={grammar_config.phi_min_deg}, "
        f"phi_step={grammar_config.phi_step_deg}, "
        f"max_run={grammar_config.max_concentrated_run}, "
        f"max_cap_plies={grammar_config.max_total_cap_plies}"
    )
    print(
        "frontier="
        f"max_states_per_depth={args.max_states_per_depth}, "
        f"max_frontier_size={args.max_frontier_size}, "
        f"log_ei_bin_width={args.log_ei_bin_width}"
    )
    print(
        "span_case="
        f"half_span={args.half_span_m:.2f}m, "
        f"root_moment={args.root_moment_Nm:.1f}Nm, "
        f"delta_allow={args.delta_allow_m:.3f}m"
    )
    print(
        "transition="
        f"monotone_cap_plies={args.enforce_monotone_cap_plies}, "
        f"common_phi_nonincreasing={args.require_common_phi_nonincreasing}, "
        f"max_drop={args.max_cap_ply_drop_per_transition}, "
        f"max_phi_change={args.max_phi_change_deg_per_transition}"
    )
    print()


def print_variant_summary(results: list[VariantResult]) -> None:
    print("variant build summary")
    header = f"{'variant':>10}  {'total_time[s]':>13}  {'frontier_sizes':>24}"
    print(header)
    print("-" * len(header))
    for result in results:
        total_time = sum(result.build_times_by_diameter.values())
        sizes = ",".join(str(len(frontier)) for frontier in result.frontiers_by_diameter.values())
        print(f"{result.config.name:>10}  {total_time:13.3f}  {sizes:>24}")
    print()


def print_frontier_comparisons(
    baseline: VariantResult,
    variants: list[VariantResult],
    diameters: tuple[float, ...],
) -> None:
    for variant in variants:
        print(f"frontier delta vs baseline: {variant.config.name}")
        header = (
            f"{'D[mm]':>7}  {'base':>6}  {'var':>6}  {'miss':>6}  {'add':>6}  "
            f"{'retain':>8}  {'time x':>8}  {'W base':>17}  {'W var':>17}  "
            f"{'EI_v base':>21}  {'EI_v var':>21}"
        )
        print(header)
        print("-" * len(header))
        for diameter in diameters:
            item = compare_diameter(baseline, variant, diameter)
            time_ratio = safe_ratio(item.variant_time_s, item.baseline_time_s)
            print(
                f"{item.diameter_mm:7.1f}  "
                f"{item.baseline_size:6d}  "
                f"{item.variant_size:6d}  "
                f"{item.missing_from_baseline:6d}  "
                f"{item.added_vs_baseline:6d}  "
                f"{item.retained_baseline_fraction:8.3f}  "
                f"{time_ratio:8.2f}  "
                f"{format_range(item.baseline_weight_range, '.4f'):>17}  "
                f"{format_range(item.variant_weight_range, '.4f'):>17}  "
                f"{format_range(item.baseline_ei_vertical_range, '.3e'):>21}  "
                f"{format_range(item.variant_ei_vertical_range, '.3e'):>21}"
            )
        print()


def compare_diameter(
    baseline: VariantResult,
    variant: VariantResult,
    diameter: float,
) -> DiameterComparison:
    key = round(float(diameter), 6)
    base = baseline.frontiers_by_diameter[key]
    var = variant.frontiers_by_diameter[key]
    base_keys = {candidate_key(candidate) for candidate in base}
    var_keys = {candidate_key(candidate) for candidate in var}
    retained = len(base_keys & var_keys)
    return DiameterComparison(
        diameter_mm=float(diameter),
        baseline_size=len(base),
        variant_size=len(var),
        baseline_time_s=baseline.build_times_by_diameter[key],
        variant_time_s=variant.build_times_by_diameter[key],
        missing_from_baseline=len(base_keys - var_keys),
        added_vs_baseline=len(var_keys - base_keys),
        retained_baseline_fraction=safe_ratio(retained, len(base_keys)),
        baseline_weight_range=value_range(c.section.weight_kg_m for c in base),
        variant_weight_range=value_range(c.section.weight_kg_m for c in var),
        baseline_ei_vertical_range=value_range(c.section.EI_vertical_Nmm2 for c in base),
        variant_ei_vertical_range=value_range(c.section.EI_vertical_Nmm2 for c in var),
    )


def print_span_comparison(
    baseline: VariantResult,
    results: list[VariantResult],
) -> None:
    base = baseline.span_result
    print("span optimizer comparison")
    header = (
        f"{'variant':>10}  {'feasible':>8}  {'W_total[kg]':>12}  "
        f"{'dW[kg]':>10}  {'delta[mm]':>10}  {'margin[mm]':>11}  {'reason'}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        span = result.span_result
        d_weight = span.total_weight_kg - base.total_weight_kg
        margin_mm = (result.span_result.deflection_tip_m - base.deflection_tip_m) * 1000.0
        print(
            f"{result.config.name:>10}  "
            f"{str(span.feasible):>8}  "
            f"{span.total_weight_kg:12.4f}  "
            f"{d_weight:10.4f}  "
            f"{span.deflection_tip_m * 1000.0:10.2f}  "
            f"{margin_mm:11.2f}  "
            f"{span.reason}"
        )

    print()
    print("selected cap_phis by station")
    for result in results:
        sequences = " | ".join(format_cap_sequence(choice.cap_phis) for choice in result.span_result.choices)
        print(f"{result.config.name:>10}: {sequences if sequences else '-'}")


def write_json_report(
    args: argparse.Namespace,
    diameters: tuple[float, ...],
    grammar_config: LaminateGrammarConfig,
    results: list[VariantResult],
    path: Path,
) -> None:
    baseline = results[0]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "diameters_mm": list(diameters),
            "grammar_config": grammar_config.__dict__,
            "frontier": {
                "max_states_per_depth": args.max_states_per_depth,
                "max_frontier_size": args.max_frontier_size,
                "log_ei_bin_width": args.log_ei_bin_width,
                "max_d_over_t": args.max_d_over_t,
            },
            "span_case": {
                "half_span_m": args.half_span_m,
                "root_moment_Nm": args.root_moment_Nm,
                "moment_power": args.moment_power,
                "delta_allow_m": args.delta_allow_m,
                "sigma_allow_MPa": args.sigma_allow_MPa,
                "min_foreaft_ratio": args.min_foreaft_ratio,
            },
            "transition": {
                "enforce_monotone_cap_plies": args.enforce_monotone_cap_plies,
                "require_common_phi_nonincreasing": args.require_common_phi_nonincreasing,
                "max_cap_ply_drop_per_transition": args.max_cap_ply_drop_per_transition,
                "max_phi_change_deg_per_transition": args.max_phi_change_deg_per_transition,
            },
        },
        "variants": [variant_report(result, baseline, diameters) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"json report -> {path}")


def variant_report(
    result: VariantResult,
    baseline: VariantResult,
    diameters: tuple[float, ...],
) -> dict[str, Any]:
    span = result.span_result
    base_span = baseline.span_result
    return {
        "name": result.config.name,
        "approximate_prefix_pruning": result.config.approximate_prefix_pruning,
        "approximate_bin_pruning": result.config.approximate_bin_pruning,
        "total_build_time_s": sum(result.build_times_by_diameter.values()),
        "diameters": [
            diameter_report(baseline, result, diameter)
            for diameter in diameters
        ],
        "span_result": {
            "feasible": span.feasible,
            "total_weight_kg": span.total_weight_kg,
            "delta_weight_vs_baseline_kg": span.total_weight_kg - base_span.total_weight_kg,
            "deflection_tip_m": span.deflection_tip_m,
            "delta_deflection_vs_baseline_m": span.deflection_tip_m - base_span.deflection_tip_m,
            "reason": span.reason,
            "selected_cap_phis": [list(choice.cap_phis) for choice in span.choices],
        },
    }


def diameter_report(
    baseline: VariantResult,
    result: VariantResult,
    diameter: float,
) -> dict[str, Any]:
    key = round(float(diameter), 6)
    frontier = result.frontiers_by_diameter[key]
    item = compare_diameter(baseline, result, diameter)
    return {
        "diameter_mm": float(diameter),
        "frontier_size": len(frontier),
        "build_time_s": result.build_times_by_diameter[key],
        "missing_from_baseline": item.missing_from_baseline,
        "added_vs_baseline": item.added_vs_baseline,
        "retained_baseline_fraction": item.retained_baseline_fraction,
        "weight_range_kg_m": list(item.variant_weight_range) if item.variant_weight_range else None,
        "ei_vertical_range_Nmm2": list(item.variant_ei_vertical_range) if item.variant_ei_vertical_range else None,
        "representative_cap_phis": [
            list(seq)
            for seq in representative_cap_phis(frontier, limit=5)
        ],
    }


def representative_cap_phis(
    candidates: list[SectionCandidate],
    limit: int,
) -> tuple[tuple[int, ...], ...]:
    if limit <= 0 or not candidates:
        return tuple()
    ordered = sorted(candidates, key=lambda c: (c.section.weight_kg_m, c.section.EI_vertical_Nmm2))
    if len(ordered) == 1:
        return (ordered[0].cap_phis,)
    count = min(limit, len(ordered))
    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    sequences: list[tuple[int, ...]] = []
    for index in indices:
        seq = ordered[index].cap_phis
        if seq not in sequences:
            sequences.append(seq)
    return tuple(sequences)


def candidate_key(candidate: SectionCandidate) -> tuple[int, ...]:
    return candidate.cap_phis


def parse_diameters(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.replace(",", " ").split())
    if not values:
        raise ValueError("--diameters must contain at least one value")
    return values


def value_range(values) -> tuple[float, float] | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return (min(finite), max(finite))


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-30:
        return float("inf")
    return float(numerator) / float(denominator)


def format_diameters(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:.1f}" for value in values)


def format_range(value_range_: tuple[float, float] | None, fmt: str) -> str:
    if value_range_ is None:
        return "-"
    lo, hi = value_range_
    return f"{lo:{fmt}}..{hi:{fmt}}"


def format_cap_sequence(sequence: tuple[int, ...]) -> str:
    if not sequence:
        return "base"
    return "/".join(f"Phi{phi}" for phi in sequence)


if __name__ == "__main__":
    main()
