from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig
from src.core.section_calculator import SectionCalculator
from src.core.section_frontier import FrontierConfig, SectionCandidate, SectionFrontierBuilder


@dataclass(frozen=True)
class FrontierStats:
    diameter_mm: float
    frontier_size: int
    elapsed_s: float
    weight_range_kg_m: tuple[float, float] | None
    ei_vertical_range_Nmm2: tuple[float, float] | None
    ei_foreaft_range_Nmm2: tuple[float, float] | None
    d_over_t_range: tuple[float, float] | None
    representative_cap_phis: tuple[tuple[int, ...], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose SectionFrontierBuilder runtime and output frontier ranges "
            "for a real-scale mandrel diameter distribution."
        )
    )
    parser.add_argument(
        "--diameters",
        default=None,
        help=(
            "Comma or space separated mandrel diameters in mm. "
            "Default is 8 points from 120 to 45 mm."
        ),
    )
    parser.add_argument("--diameter-root-mm", type=float, default=120.0)
    parser.add_argument("--diameter-tip-mm", type=float, default=45.0)
    parser.add_argument("--n-diameters", type=int, default=8)
    parser.add_argument("--phi-min-deg", type=int, default=20)
    parser.add_argument("--phi-step-deg", type=int, default=20)
    parser.add_argument(
        "--max-total-cap-plies",
        type=int,
        default=4,
        help="Light default for first real-scale diagnosis. Increase toward 18 for aero config parity.",
    )
    parser.add_argument("--max-concentrated-run", type=int, default=10)
    parser.add_argument("--max-states-per-depth", type=int, default=400)
    parser.add_argument("--max-frontier-size", type=int, default=800)
    parser.add_argument("--max-d-over-t", type=float, default=None)
    parser.add_argument("--min-ei-vertical", type=float, default=0.0)
    parser.add_argument("--min-ei-foreaft", type=float, default=0.0)
    parser.add_argument("--log-ei-bin-width", type=float, default=0.015)
    parser.add_argument("--approx-prefix-pruning", action="store_true")
    parser.add_argument("--approx-bin-pruning", action="store_true")
    parser.add_argument("--representatives", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diameters = _diameters_from_args(args)
    grammar_config = LaminateGrammarConfig(
        phi_min_deg=args.phi_min_deg,
        phi_max_deg=180,
        phi_step_deg=args.phi_step_deg,
        max_concentrated_run=args.max_concentrated_run,
        max_total_cap_plies=args.max_total_cap_plies,
    )
    frontier_config = FrontierConfig(
        log_ei_bin_width=args.log_ei_bin_width,
        max_states_per_depth=args.max_states_per_depth,
        max_frontier_size=args.max_frontier_size,
        max_d_over_t=args.max_d_over_t,
        min_ei_vertical_Nmm2=args.min_ei_vertical,
        min_ei_foreaft_Nmm2=args.min_ei_foreaft,
        approximate_prefix_pruning=args.approx_prefix_pruning,
        approximate_bin_pruning=args.approx_bin_pruning,
    )
    builder = SectionFrontierBuilder(
        grammar=LaminateGrammar(grammar_config),
        calculator=SectionCalculator(),
        config=frontier_config,
    )

    print_config(diameters, grammar_config, frontier_config)
    stats = [diagnose_diameter(builder, diameter, args.representatives) for diameter in diameters]
    print_table(stats)
    print_summary(stats)


def diagnose_diameter(
    builder: SectionFrontierBuilder,
    diameter_mm: float,
    representative_count: int,
) -> FrontierStats:
    start = time.perf_counter()
    frontier = builder.build(diameter_mm)
    elapsed = time.perf_counter() - start
    return FrontierStats(
        diameter_mm=diameter_mm,
        frontier_size=len(frontier),
        elapsed_s=elapsed,
        weight_range_kg_m=_range(c.section.weight_kg_m for c in frontier),
        ei_vertical_range_Nmm2=_range(c.section.EI_vertical_Nmm2 for c in frontier),
        ei_foreaft_range_Nmm2=_range(c.section.EI_foreaft_Nmm2 for c in frontier),
        d_over_t_range=_range(c.section.D_over_t_conservative for c in frontier),
        representative_cap_phis=representative_cap_phis(frontier, representative_count),
    )


def print_config(
    diameters: tuple[float, ...],
    grammar_config: LaminateGrammarConfig,
    frontier_config: FrontierConfig,
) -> None:
    print("frontier runtime diagnosis")
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
        f"max_states_per_depth={frontier_config.max_states_per_depth}, "
        f"max_frontier_size={frontier_config.max_frontier_size}, "
        f"approx_prefix={frontier_config.approximate_prefix_pruning}, "
        f"approx_bin={frontier_config.approximate_bin_pruning}"
    )
    print()


def print_table(stats: list[FrontierStats]) -> None:
    header = (
        f"{'D[mm]':>7}  {'frontier':>8}  {'time[s]':>8}  "
        f"{'W[kg/m]':>17}  {'EI_v[Nmm2]':>21}  {'EI_f[Nmm2]':>21}  "
        f"{'D/t':>13}  cap_phis"
    )
    print(header)
    print("-" * len(header))
    for item in stats:
        print(
            f"{item.diameter_mm:7.1f}  "
            f"{item.frontier_size:8d}  "
            f"{item.elapsed_s:8.3f}  "
            f"{format_range(item.weight_range_kg_m, '.4f'):>17}  "
            f"{format_range(item.ei_vertical_range_Nmm2, '.3e'):>21}  "
            f"{format_range(item.ei_foreaft_range_Nmm2, '.3e'):>21}  "
            f"{format_range(item.d_over_t_range, '.1f'):>13}  "
            f"{format_cap_sequences(item.representative_cap_phis)}"
        )


def print_summary(stats: list[FrontierStats]) -> None:
    total_time = sum(item.elapsed_s for item in stats)
    largest = max(stats, key=lambda item: item.frontier_size, default=None)
    slowest = max(stats, key=lambda item: item.elapsed_s, default=None)
    print()
    print(f"total_time_s={total_time:.3f}")
    if largest is not None:
        print(f"largest_frontier=D{largest.diameter_mm:.1f}mm size={largest.frontier_size}")
    if slowest is not None:
        print(f"slowest_build=D{slowest.diameter_mm:.1f}mm time_s={slowest.elapsed_s:.3f}")


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


def _diameters_from_args(args: argparse.Namespace) -> tuple[float, ...]:
    if args.diameters:
        values = [float(part) for part in args.diameters.replace(",", " ").split()]
        if not values:
            raise ValueError("--diameters did not contain any numeric values")
        return tuple(values)

    if args.n_diameters <= 0:
        raise ValueError("--n-diameters must be positive")
    if args.n_diameters == 1:
        return (float(args.diameter_root_mm),)

    step = (args.diameter_tip_mm - args.diameter_root_mm) / (args.n_diameters - 1)
    return tuple(args.diameter_root_mm + step * i for i in range(args.n_diameters))


def _range(values) -> tuple[float, float] | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return (min(finite), max(finite))


def format_diameters(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:.1f}" for value in values)


def format_range(value_range: tuple[float, float] | None, fmt: str) -> str:
    if value_range is None:
        return "-"
    lo, hi = value_range
    return f"{lo:{fmt}}..{hi:{fmt}}"


def format_cap_sequences(sequences: tuple[tuple[int, ...], ...]) -> str:
    if not sequences:
        return "-"
    return " | ".join(format_cap_sequence(sequence) for sequence in sequences)


def format_cap_sequence(sequence: tuple[int, ...]) -> str:
    if not sequence:
        return "base"
    return "/".join(f"Phi{phi}" for phi in sequence)


if __name__ == "__main__":
    main()
