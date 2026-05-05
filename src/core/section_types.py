from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from math import pi
from typing import Optional


G0 = 9.80665


class LayerKind(str, Enum):
    INNER_90 = "inner_90"
    OUTER_90 = "outer_90"
    BIAS_PLUS_45 = "bias_plus_45"
    BIAS_MINUS_45 = "bias_minus_45"
    CAP_0 = "cap_0"


@dataclass(frozen=True)
class Material:
    name: str
    young_modulus_N_mm2: float
    density_kg_m3: float
    ply_thickness_mm: float


@dataclass(frozen=True)
class MaterialLibrary:
    inner_outer_90: Material
    bias_45: Material
    cap_0: Material

    @staticmethod
    def default_hpa() -> "MaterialLibrary":
        return MaterialLibrary(
            inner_outer_90=Material(
                name="24t_90",
                young_modulus_N_mm2=900.0 * G0,
                density_kg_m3=1496.0,
                ply_thickness_mm=0.125,
            ),
            bias_45=Material(
                name="40t_45",
                young_modulus_N_mm2=1900.0 * G0,
                density_kg_m3=1559.0,
                ply_thickness_mm=0.111,
            ),
            cap_0=Material(
                name="40t_0",
                young_modulus_N_mm2=22000.0 * G0,
                density_kg_m3=1559.0,
                ply_thickness_mm=0.111,
            ),
        )


@dataclass(frozen=True)
class LayerSpec:
    kind: LayerKind
    phi_deg: Optional[float] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class LaminateStack:
    layers: tuple[LayerSpec, ...]

    def cap_phis(self) -> tuple[float, ...]:
        return tuple(
            float(layer.phi_deg)
            for layer in self.layers
            if layer.kind == LayerKind.CAP_0 and layer.phi_deg is not None
        )

    def layer_count(self) -> int:
        return len(self.layers)

    def cap_layer_count(self) -> int:
        return sum(1 for layer in self.layers if layer.kind == LayerKind.CAP_0)


@dataclass(frozen=True)
class ResolvedLayer:
    index: int
    kind: LayerKind
    material: Material
    thickness_mm: float
    phi_deg: float
    label: str


@dataclass(frozen=True)
class AngularPatch:
    layer_index: int
    kind: LayerKind
    material: Material
    theta_start_rad: float
    theta_end_rad: float
    r_inner_mm: float
    r_outer_mm: float


@dataclass(frozen=True)
class StressResult:
    max_abs_stress_MPa: float
    critical_layer_index: int
    critical_kind: LayerKind
    critical_theta_rad: float
    critical_radius_mm: float


@dataclass(frozen=True)
class SectionCalculatorConfig:
    phi_min_deg: float = 0.0
    phi_max_deg: float = 180.0
    eps: float = 1e-9
    include_debug_patches: bool = True


@dataclass(frozen=True)
class SectionResult:
    mandrel_diameter_mm: float
    stack: LaminateStack
    patches: tuple[AngularPatch, ...]
    area_mm2: float
    weight_kg_m: float
    EI_vertical_Nmm2: float
    EI_foreaft_Nmm2: float
    I_vertical_geom_mm4: float
    I_foreaft_geom_mm4: float
    t_min_mm: float
    t_max_mm: float
    D_outer_min_mm: float
    D_outer_max_mm: float
    D_over_t_conservative: float
    layer_count_total: int
    cap_layer_count: int

    @property
    def EI_vertical_kgfmm2(self) -> float:
        return self.EI_vertical_Nmm2 / G0

    @property
    def EI_foreaft_kgfmm2(self) -> float:
        return self.EI_foreaft_Nmm2 / G0

    def max_stress_vertical(self, moment_Nmm: float) -> StressResult:
        if self.EI_vertical_Nmm2 <= 0.0:
            return StressResult(
                max_abs_stress_MPa=float("inf"),
                critical_layer_index=-1,
                critical_kind=LayerKind.CAP_0,
                critical_theta_rad=0.0,
                critical_radius_mm=0.0,
            )

        curvature_1_mm = moment_Nmm / self.EI_vertical_Nmm2
        best_abs = -1.0
        best_layer = -1
        best_kind = LayerKind.CAP_0
        best_theta = 0.0
        best_radius = 0.0

        for patch in self.patches:
            theta = _theta_for_max_abs_cos(patch.theta_start_rad, patch.theta_end_rad)
            z_mm = patch.r_outer_mm * abs(math.cos(theta))
            stress = patch.material.young_modulus_N_mm2 * curvature_1_mm * z_mm
            abs_stress = abs(stress)
            if abs_stress > best_abs:
                best_abs = abs_stress
                best_layer = patch.layer_index
                best_kind = patch.kind
                best_theta = theta
                best_radius = patch.r_outer_mm

        return StressResult(
            max_abs_stress_MPa=best_abs,
            critical_layer_index=best_layer,
            critical_kind=best_kind,
            critical_theta_rad=best_theta,
            critical_radius_mm=best_radius,
        )


def _theta_for_max_abs_cos(theta_start: float, theta_end: float) -> float:
    candidates = [theta_start, theta_end]
    two_pi = 2.0 * pi
    for base in (0.0, pi):
        k_start = int((theta_start - base) // two_pi) - 1
        k_end = int((theta_end - base) // two_pi) + 2
        for k in range(k_start, k_end + 1):
            theta = base + k * two_pi
            if theta_start <= theta <= theta_end:
                candidates.append(theta)
    return max(candidates, key=lambda x: abs(math.cos(x)))
