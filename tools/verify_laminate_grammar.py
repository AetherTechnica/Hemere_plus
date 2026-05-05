from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.laminate_grammar import LaminateGrammar, LaminateGrammarConfig


def expect_valid(grammar: LaminateGrammar, phis: tuple[int, ...]) -> None:
    if not grammar.is_valid_cap_sequence(phis):
        raise AssertionError(f"expected valid sequence: {phis}")


def expect_invalid(grammar: LaminateGrammar, phis: tuple[int, ...]) -> None:
    if grammar.is_valid_cap_sequence(phis):
        raise AssertionError(f"expected invalid sequence: {phis}")


def main() -> None:
    grammar = LaminateGrammar(LaminateGrammarConfig(phi_step_deg=10, max_total_cap_plies=40))

    expect_valid(grammar, (50, 40, 30, 20))
    expect_invalid(grammar, (50, 40, 60))
    expect_valid(grammar, (170, 160, 140, 100, 90, 80, 70, 60, 50, 40, 180, 100, 70, 50, 30, 180, 100, 70, 50, 30))
    expect_invalid(grammar, (170, 160, 140, 100, 90, 80, 70, 60, 50, 40, 30))
    expect_valid(grammar, (170, 160, 140, 100, 90, 80, 70, 60, 50, 40, 180, 170))

    stack = grammar.build_stack((50, 40, 30, 20))
    assert stack.layers[0].kind.value == "inner_90"
    assert stack.layers[-1].kind.value == "outer_90"
    assert stack.cap_phis() == (50.0, 40.0, 30.0, 20.0)

    print("laminate grammar verification complete")


if __name__ == "__main__":
    main()
