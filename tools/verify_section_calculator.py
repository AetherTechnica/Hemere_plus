from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.section_calculator import SectionCalculator
from src.core.section_types import LaminateStack, LayerKind, LayerSpec, MaterialLibrary


def assert_close(actual: float, expected: float, rel: float = 1e-9, label: str = "") -> None:
    scale = max(abs(expected), 1.0)
    err = abs(actual - expected) / scale
    if err > rel:
        raise AssertionError(f"{label}: actual={actual:.12e}, expected={expected:.12e}, rel_err={err:.3e}")


def closed_form(diameter_mm: float, thickness_mm: float, phi_deg: float):
    phi = math.radians(phi_deg)
    di = diameter_mm
    do = diameter_mm + 2.0 * thickness_mm
    d2 = do**2 - di**2
    d4 = do**4 - di**4
    area = d2 * phi / 4.0
    i_vertical = d4 * (phi + math.sin(phi)) / 64.0
    i_foreaft = d4 * (phi - math.sin(phi)) / 64.0
    return area, i_vertical, i_foreaft


def verify_closed_form() -> None:
    calc = SectionCalculator()
    mat = MaterialLibrary.default_hpa().cap_0
    diameter = 100.0
    for phi in (20.0, 60.0, 120.0, 180.0):
        stack = LaminateStack((LayerSpec(LayerKind.CAP_0, phi_deg=phi),))
        result = calc.evaluate(diameter, stack)
        area, iv, ifa = closed_form(diameter, mat.ply_thickness_mm, phi)
        assert_close(result.area_mm2, area, label=f"area Phi{phi}")
        assert_close(result.I_vertical_geom_mm4, iv, label=f"Iv Phi{phi}")
        assert_close(result.I_foreaft_geom_mm4, ifa, label=f"If Phi{phi}")
    print("closed-form section checks: PASS")


def verify_monotonicity() -> None:
    calc = SectionCalculator()
    base = LaminateStack((
        LayerSpec(LayerKind.INNER_90),
        LayerSpec(LayerKind.BIAS_PLUS_45),
        LayerSpec(LayerKind.BIAS_MINUS_45),
        LayerSpec(LayerKind.OUTER_90),
    ))
    plus = LaminateStack(base.layers[:-1] + (LayerSpec(LayerKind.CAP_0, phi_deg=180.0),) + base.layers[-1:])
    r0 = calc.evaluate(90.0, base)
    r1 = calc.evaluate(90.0, plus)
    assert r1.weight_kg_m > r0.weight_kg_m
    assert r1.EI_vertical_Nmm2 > r0.EI_vertical_Nmm2
    assert r1.EI_foreaft_Nmm2 > r0.EI_foreaft_Nmm2
    print("monotonicity checks: PASS")


def verify_thickness_split() -> None:
    calc = SectionCalculator()
    stack = LaminateStack((
        LayerSpec(LayerKind.INNER_90),
        LayerSpec(LayerKind.CAP_0, phi_deg=20.0),
        LayerSpec(LayerKind.OUTER_90),
    ))
    result = calc.evaluate(80.0, stack)
    assert result.t_max_mm > result.t_min_mm
    print("non-uniform thickness checks: PASS")


def verify_stress() -> None:
    calc = SectionCalculator()
    mat = MaterialLibrary.default_hpa().cap_0
    diameter = 100.0
    stack = LaminateStack((LayerSpec(LayerKind.CAP_0, phi_deg=180.0),))
    result = calc.evaluate(diameter, stack)
    moment = 1.0e6
    stress = result.max_stress_vertical(moment).max_abs_stress_MPa
    _, iv, _ = closed_form(diameter, mat.ply_thickness_mm, 180.0)
    c = (diameter + 2.0 * mat.ply_thickness_mm) / 2.0
    expected = moment * c / iv
    assert_close(stress, expected, rel=1e-9, label="full pipe stress")
    print("stress checks: PASS")


def main() -> None:
    verify_closed_form()
    verify_monotonicity()
    verify_thickness_split()
    verify_stress()
    print("section calculator verification complete")


if __name__ == "__main__":
    main()
