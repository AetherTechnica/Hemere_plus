from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.section_types import (
    AngularPatch,
    LaminateStack,
    LayerKind,
    LayerSpec,
    Material,
    MaterialLibrary,
    ResolvedLayer,
    SectionCalculatorConfig,
    SectionResult,
)


TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class _RadiusSegment:
    theta_start_rad: float
    theta_end_rad: float
    radius_mm: float


class SectionCalculator:
    def __init__(
        self,
        material_library: MaterialLibrary | None = None,
        config: SectionCalculatorConfig | None = None,
    ):
        self.materials = material_library or MaterialLibrary.default_hpa()
        self.config = config or SectionCalculatorConfig()

    def evaluate(self, mandrel_diameter_mm: float, stack: LaminateStack) -> SectionResult:
        if mandrel_diameter_mm <= 0.0:
            raise ValueError("mandrel_diameter_mm must be positive")

        r0 = mandrel_diameter_mm / 2.0
        segments = (_RadiusSegment(0.0, TWO_PI, r0),)
        patches: list[AngularPatch] = []
        area_mm2 = 0.0
        weight_kg_m = 0.0
        ei_vertical = 0.0
        ei_foreaft = 0.0
        i_vertical = 0.0
        i_foreaft = 0.0

        for layer in self._resolve_layers(stack):
            coverage = self._coverage_intervals(layer.phi_deg)
            segments, new_patches = self._apply_layer(segments, coverage, layer)
            patches.extend(new_patches)

            for patch in new_patches:
                props = self._patch_properties(patch)
                area_mm2 += props["area_mm2"]
                weight_kg_m += props["weight_kg_m"]
                i_vertical += props["I_vertical_mm4"]
                i_foreaft += props["I_foreaft_mm4"]
                ei_vertical += props["EI_vertical_Nmm2"]
                ei_foreaft += props["EI_foreaft_Nmm2"]

        final_radii = [segment.radius_mm for segment in segments]
        t_values = [radius - r0 for radius in final_radii]
        t_min = min(t_values) if t_values else 0.0
        t_max = max(t_values) if t_values else 0.0
        d_min = 2.0 * min(final_radii) if final_radii else mandrel_diameter_mm
        d_max = 2.0 * max(final_radii) if final_radii else mandrel_diameter_mm
        d_over_t = float("inf") if t_min <= self.config.eps else d_max / t_min

        return SectionResult(
            mandrel_diameter_mm=float(mandrel_diameter_mm),
            stack=stack,
            patches=tuple(patches) if self.config.include_debug_patches else tuple(),
            area_mm2=float(area_mm2),
            weight_kg_m=float(weight_kg_m),
            EI_vertical_Nmm2=float(ei_vertical),
            EI_foreaft_Nmm2=float(ei_foreaft),
            I_vertical_geom_mm4=float(i_vertical),
            I_foreaft_geom_mm4=float(i_foreaft),
            t_min_mm=float(t_min),
            t_max_mm=float(t_max),
            D_outer_min_mm=float(d_min),
            D_outer_max_mm=float(d_max),
            D_over_t_conservative=float(d_over_t),
            layer_count_total=stack.layer_count(),
            cap_layer_count=stack.cap_layer_count(),
        )

    def _resolve_layers(self, stack: LaminateStack) -> tuple[ResolvedLayer, ...]:
        resolved: list[ResolvedLayer] = []
        for idx, spec in enumerate(stack.layers):
            material = self._material_for(spec)
            phi = self._phi_for(spec)
            if phi <= self.config.phi_min_deg or phi > self.config.phi_max_deg:
                raise ValueError(f"invalid phi_deg={phi} for layer {idx}")
            resolved.append(
                ResolvedLayer(
                    index=idx,
                    kind=spec.kind,
                    material=material,
                    thickness_mm=material.ply_thickness_mm,
                    phi_deg=float(phi),
                    label=spec.label or spec.kind.value,
                )
            )
        return tuple(resolved)

    def _material_for(self, layer: LayerSpec) -> Material:
        if layer.kind in (LayerKind.INNER_90, LayerKind.OUTER_90):
            return self.materials.inner_outer_90
        if layer.kind in (LayerKind.BIAS_PLUS_45, LayerKind.BIAS_MINUS_45):
            return self.materials.bias_45
        if layer.kind == LayerKind.CAP_0:
            return self.materials.cap_0
        raise ValueError(f"unsupported layer kind: {layer.kind}")

    def _phi_for(self, layer: LayerSpec) -> float:
        if layer.kind == LayerKind.CAP_0:
            if layer.phi_deg is None:
                raise ValueError("CAP_0 layers require phi_deg")
            return float(layer.phi_deg)
        if layer.phi_deg is not None:
            raise ValueError(f"{layer.kind.value} must not set phi_deg")
        return 180.0

    def _coverage_intervals(self, phi_deg: float) -> tuple[tuple[float, float], ...]:
        if phi_deg >= 180.0 - self.config.eps:
            return ((0.0, TWO_PI),)

        phi = math.radians(phi_deg)
        half = phi / 2.0
        return (
            (0.0, half),
            (TWO_PI - half, TWO_PI),
            (math.pi - half, math.pi + half),
        )

    def _apply_layer(
        self,
        segments: tuple[_RadiusSegment, ...],
        coverage: tuple[tuple[float, float], ...],
        layer: ResolvedLayer,
    ) -> tuple[tuple[_RadiusSegment, ...], list[AngularPatch]]:
        boundaries = {0.0, TWO_PI}
        for segment in segments:
            boundaries.add(segment.theta_start_rad)
            boundaries.add(segment.theta_end_rad)
        for start, end in coverage:
            boundaries.add(start)
            boundaries.add(end)

        sorted_bounds = sorted(boundaries)
        new_segments: list[_RadiusSegment] = []
        patches: list[AngularPatch] = []

        for start, end in zip(sorted_bounds[:-1], sorted_bounds[1:]):
            if end - start <= self.config.eps:
                continue
            mid = (start + end) / 2.0
            radius = self._radius_at(segments, mid)
            covered = self._is_covered(coverage, mid)
            if covered:
                r_outer = radius + layer.thickness_mm
                patches.append(
                    AngularPatch(
                        layer_index=layer.index,
                        kind=layer.kind,
                        material=layer.material,
                        theta_start_rad=start,
                        theta_end_rad=end,
                        r_inner_mm=radius,
                        r_outer_mm=r_outer,
                    )
                )
                new_segments.append(_RadiusSegment(start, end, r_outer))
            else:
                new_segments.append(_RadiusSegment(start, end, radius))

        return tuple(self._merge_segments(new_segments)), patches

    def _radius_at(self, segments: tuple[_RadiusSegment, ...], theta: float) -> float:
        for segment in segments:
            if segment.theta_start_rad - self.config.eps <= theta <= segment.theta_end_rad + self.config.eps:
                return segment.radius_mm
        raise RuntimeError(f"no radius segment covers theta={theta}")

    def _is_covered(self, coverage: tuple[tuple[float, float], ...], theta: float) -> bool:
        return any(start - self.config.eps <= theta <= end + self.config.eps for start, end in coverage)

    def _merge_segments(self, segments: list[_RadiusSegment]) -> list[_RadiusSegment]:
        if not segments:
            return []
        merged = [segments[0]]
        for segment in segments[1:]:
            prev = merged[-1]
            if abs(prev.radius_mm - segment.radius_mm) <= self.config.eps and abs(prev.theta_end_rad - segment.theta_start_rad) <= self.config.eps:
                merged[-1] = _RadiusSegment(prev.theta_start_rad, segment.theta_end_rad, prev.radius_mm)
            else:
                merged.append(segment)
        return merged

    def _patch_properties(self, patch: AngularPatch) -> dict[str, float]:
        start = patch.theta_start_rad
        end = patch.theta_end_rad
        angle = end - start
        r2_diff = patch.r_outer_mm**2 - patch.r_inner_mm**2
        r4_diff = patch.r_outer_mm**4 - patch.r_inner_mm**4

        int_cos2 = angle / 2.0 + (math.sin(2.0 * end) - math.sin(2.0 * start)) / 4.0
        int_sin2 = angle / 2.0 - (math.sin(2.0 * end) - math.sin(2.0 * start)) / 4.0

        area = 0.5 * r2_diff * angle
        i_vertical = 0.25 * r4_diff * int_cos2
        i_foreaft = 0.25 * r4_diff * int_sin2

        return {
            "area_mm2": area,
            "weight_kg_m": area * patch.material.density_kg_m3 * 1e-6,
            "I_vertical_mm4": i_vertical,
            "I_foreaft_mm4": i_foreaft,
            "EI_vertical_Nmm2": i_vertical * patch.material.young_modulus_N_mm2,
            "EI_foreaft_Nmm2": i_foreaft * patch.material.young_modulus_N_mm2,
        }
