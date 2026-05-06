from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.exact_section_enumerator import ExactEnumerationConfig
from src.core.laminate_grammar import LaminateGrammarConfig
from src.core.section_design_table import SectionDesignTableConfig
from src.integration.aero_struct_optimizer import (
    AeroStructDesignParams,
    AeroStructOptimizer,
    AeroStructOptimizerConfig,
)
from tools.build_exact_section_tables import build_one_diameter, filename_for


def main() -> None:
    grammar_config = LaminateGrammarConfig(
        phi_min_deg=60,
        phi_max_deg=180,
        phi_step_deg=60,
        max_concentrated_run=2,
        max_total_cap_plies=3,
    )
    table_config = SectionDesignTableConfig(log_ei_bin_width=0.05)
    diameters = (120.0, 80.0)

    with tempfile.TemporaryDirectory() as tmp:
        table_dir = Path(tmp)
        files = []
        for diameter in diameters:
            payload = build_one_diameter(
                diameter_mm=diameter,
                grammar_config=grammar_config,
                enumeration_config=ExactEnumerationConfig(),
                table_config=table_config,
                save_all_candidates=False,
                progress_every=0,
                include_debug_patches=False,
            )
            filename = filename_for(diameter, grammar_config, table_config)
            (table_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            files.append(filename)

        manifest = {
            "diameters_mm": list(diameters),
            "grammar_config": grammar_config.__dict__,
            "table_config": table_config.__dict__,
            "files": files,
        }
        manifest_path = table_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        optimizer = AeroStructOptimizer(
            AeroStructDesignParams(
                W_pilot_kg=70.0,
                W_fuselage_kg=10.0,
                W_wing_secondary_kg=3.0,
                span_m=8.0,
                V_flight_ms=8.0,
                delta_allow_m=1.0,
                sigma_allow_MPa=300.0,
                mandrel_diameters_mm=diameters,
                n_spar=len(diameters),
                n_aero=20,
            ),
            AeroStructOptimizerConfig(
                n_beta=1,
                grammar_config=grammar_config,
                section_table_manifest_path=str(manifest_path),
                max_candidates_per_station=50,
            ),
        )

        for diameter in diameters:
            candidates = optimizer._frontier_for(diameter)
            if not candidates:
                raise AssertionError(f"table lookup produced no candidates for D={diameter}")
            if any(candidate.section.patches == tuple() for candidate in candidates):
                raise AssertionError("table-backed candidates must be re-evaluated with stress patches")

        result = optimizer.optimize(verbose=False)
        if not result.sweep_results:
            raise AssertionError("table-backed optimizer produced no sweep results")
        if not result.best.span_result.choices:
            raise AssertionError("table-backed optimizer produced no span choices")

    print("aero-struct table lookup verification complete")


if __name__ == "__main__":
    main()
