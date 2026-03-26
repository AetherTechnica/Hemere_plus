"""
opt-ATLAS 3条件での統合検証スクリプト
======================================
Hemere の計算結果を opt-ATLAS の参照値と比較して精度を確認する。

検証条件:
  条件1: スパン 32.5m, 速度 7.2m/s  → beta ≈ 0.9794, Di ≈ 10.301N
  条件2: スパン 29m,   速度 8.3m/s  → beta ≈ 0.9809, Di ≈ 7.670N
  条件3: スパン 22m,   速度 11m/s   → beta ≈ 0.9888, Di ≈ 6.229N

比較方法:
  - 指定 beta での Di を計算し、opt-ATLAS の期待値と比較
  - Di は W_total に依存するため、単純比較ではなく傾向確認が主目的
  - 最適 beta 探索（スイープ）も実施
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.aerodynamics.aerodynamics_analyzer import AerodynamicsAnalyzer, AircraftAeroParams
from src.structural.structural_analyzer import StructuralAnalyzer
from src.core.ei_estimator import EIEstimator
from src.core.layup_optimizer import LayupOptimizer
from src.integration.design_integrator_v3 import DesignIntegratorV3

g = 9.80665

# ===========================================================================
# opt-ATLAS 参照値
# ===========================================================================
ATLAS_CONDITIONS = {
    1: {
        "name": "条件1 (32.5m, 7.2m/s)",
        "W_pilot_kg": 58.0, "W_fixed_other_kg": 19.0, "W_wing_fixed_kg": 9.0,
        "span_m": 32.5, "V": 7.2, "delta_allow_m": 2.2,
        "mandrel_D": np.linspace(120, 45, 8),
        "beta_opt": 0.9794, "Di_opt": 10.301,
    },
    2: {
        "name": "条件2 (29m, 8.3m/s)",
        "W_pilot_kg": 58.0, "W_fixed_other_kg": 19.0, "W_wing_fixed_kg": 9.0,
        "span_m": 29.0, "V": 8.3, "delta_allow_m": 2.0,
        "mandrel_D": np.linspace(110, 40, 8),
        "beta_opt": 0.9809, "Di_opt": 7.670,
    },
    3: {
        "name": "条件3 (22m, 11m/s)",
        "W_pilot_kg": 55.0, "W_fixed_other_kg": 20.0, "W_wing_fixed_kg": 8.0,
        "span_m": 22.0, "V": 11.0, "delta_allow_m": 0.9,
        "mandrel_D": np.linspace(90, 40, 8),
        "beta_opt": 0.9888, "Di_opt": 6.229,
    },
}

# ===========================================================================
# 1. 空力計算の検証（beta の Di 計算）
# ===========================================================================

def verify_aero_only(cond_id):
    """
    opt-ATLAS の beta_opt を固定して Di を計算し、空力計算精度を確認する。
    """
    cond  = ATLAS_CONDITIONS[cond_id]
    W_total_kg = cond['W_pilot_kg'] + cond['W_fixed_other_kg'] + cond['W_wing_fixed_kg']

    aero_params = AircraftAeroParams(
        lift_target_N=W_total_kg * g,
        span_m=cond['span_m'],
        v_flight_ms=cond['V'],
        rho_air=1.154,
        n_segments=100,
    )
    aero = AerodynamicsAnalyzer(aero_params)
    res  = aero.solve(cond['beta_opt'])
    Di   = res['induced_drag_N']

    err = (Di - cond['Di_opt']) / cond['Di_opt'] * 100
    return Di, err


# ===========================================================================
# 2. 統合フロー（空力 + 構造 + 積層最適化）
# ===========================================================================

def run_integration(cond_id, verbose=False):
    """
    指定条件で Hemere の統合フローを実行する。
    beta を opt-ATLAS の最適値に固定して検証。
    """
    cond  = ATLAS_CONDITIONS[cond_id]
    W_total_kg = cond['W_pilot_kg'] + cond['W_fixed_other_kg'] + cond['W_wing_fixed_kg']

    aero_params = AircraftAeroParams(
        lift_target_N=W_total_kg * g,
        span_m=cond['span_m'],
        v_flight_ms=cond['V'],
        rho_air=1.154,
        n_segments=8,
    )

    integrator = DesignIntegratorV3(
        aero_params,
        mandrel_diameters_mm=cond['mandrel_D']
    )

    result = integrator.optimize_full_wing(
        beta=cond['beta_opt'],
        max_deflection_m=cond['delta_allow_m'],
        verbose=verbose,
    )
    return result


# ===========================================================================
# 3. beta スイープ（Di を最小化する beta* の探索）
# ===========================================================================

def beta_sweep(cond_id, beta_range=(0.97, 1.01), n_beta=9, verbose=False):
    """
    beta をスイープして Di が最小になる beta* を求める。

    Returns:
        beta_best, Di_best, results_list
    """
    cond  = ATLAS_CONDITIONS[cond_id]
    W_total_kg = cond['W_pilot_kg'] + cond['W_fixed_other_kg'] + cond['W_wing_fixed_kg']

    aero_params = AircraftAeroParams(
        lift_target_N=W_total_kg * g,
        span_m=cond['span_m'],
        v_flight_ms=cond['V'],
        rho_air=1.154,
        n_segments=8,
    )

    integrator = DesignIntegratorV3(
        aero_params,
        mandrel_diameters_mm=cond['mandrel_D']
    )

    betas = np.linspace(beta_range[0], beta_range[1], n_beta)
    results = []
    for beta in betas:
        res = integrator.optimize_full_wing(
            beta=beta,
            max_deflection_m=cond['delta_allow_m'],
            verbose=False,
        )
        if res:
            results.append((beta, res))

    if not results:
        return None, None, []

    best = min(results, key=lambda x: x[1]['induced_drag_N'])
    return best[0], best[1]['induced_drag_N'], results


# ===========================================================================
# メイン
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Hemere Phase B 統合検証")
    print("  opt-ATLAS 3条件との精度比較")
    print("=" * 70)
    print()

    # -----------------------------------------------------------------
    # Step A: 空力計算の精度確認（構造なし、beta 固定）
    # -----------------------------------------------------------------
    print("-" * 70)
    print("Step A: 空力計算精度（beta=opt-ATLAS最適値 固定）")
    print("-" * 70)
    print(f"{'条件':^18} | {'Hemere Di [N]':>14} | {'ATLAS Di [N]':>13} | {'誤差 [%]':>10}")
    print("-" * 70)
    for cond_id in [1, 2, 3]:
        Di, err = verify_aero_only(cond_id)
        name = ATLAS_CONDITIONS[cond_id]['name']
        Di_ref = ATLAS_CONDITIONS[cond_id]['Di_opt']
        flag = "OK" if abs(err) < 2.0 else "NG!"
        print(f" {name:<18} | {Di:>13.4f} N | {Di_ref:>12.3f} N | {err:>+9.2f}% {flag}")
    print()
    print("  注: W_total の差（自重の扱い）により Di に差が出る。")
    print("  空力計算自体は verify_aero.py で 0.000% 精度検証済み。")
    print()

    # -----------------------------------------------------------------
    # Step B: 統合フロー（beta=最適値 固定）
    # -----------------------------------------------------------------
    print("-" * 70)
    print("Step B: 統合フロー（EI逆算 + 積層最適化 + 構造検証）")
    print("-" * 70)
    for cond_id in [3, 2, 1]:  # 条件3から（最もシンプル）
        cond = ATLAS_CONDITIONS[cond_id]
        print(f"\n{cond['name']}:")
        res = run_integration(cond_id, verbose=True)

        if res:
            print(f"\n  [検証結果サマリー]")
            print(f"  Di          : {res['induced_drag_N']:.4f} N  (ATLAS参照: {cond['Di_opt']:.3f} N)")
            print(f"  delta_tip   : {res['delta_tip_m']*1000:.1f} mm  (制約: {cond['delta_allow_m']*1000:.1f} mm)")
            print(f"  sigma_max   : {res['sigma_max_MPa']:.1f} MPa  (制約: 300.0 MPa)")
            print(f"  W_struct    : {res['W_struct_kg']:.3f} kg")
            print(f"  feasible    : {res['feasible']}")

            # 制約チェック
            delta_ok = res['delta_tip_m'] <= cond['delta_allow_m'] * 1.05
            sigma_ok = res['sigma_max_MPa'] <= 300.0
            print(f"  制約チェック: delta={'OK' if delta_ok else 'NG!'}, sigma={'OK' if sigma_ok else 'NG!'}")
        print()

    # -----------------------------------------------------------------
    # Step C: beta スイープ（Di 最小化）
    # -----------------------------------------------------------------
    print("-" * 70)
    print("Step C: beta スイープ (Di 最小化, 条件3のみ)")
    print("-" * 70)

    beta_best, Di_best, sweep_results = beta_sweep(
        cond_id=3, beta_range=(0.97, 1.01), n_beta=9
    )

    if sweep_results:
        print(f"  {'beta':>8} | {'Di [N]':>10} | {'delta [mm]':>12} | {'W_struct [kg]':>14} | {'可否':>6}")
        print("  " + "-" * 60)
        for beta, res in sweep_results:
            cond = ATLAS_CONDITIONS[3]
            delta_ok = res['delta_tip_m'] <= cond['delta_allow_m'] * 1.05
            mark = "OK" if (res['feasible'] and delta_ok) else "--"
            print(f"  {beta:>8.4f} | {res['induced_drag_N']:>10.4f} | "
                  f"{res['delta_tip_m']*1000:>10.1f} mm | "
                  f"{res['W_struct_kg']:>12.3f} kg | {mark:>6}")

        print()
        print(f"  最小 Di: {Di_best:.4f} N at beta={beta_best:.4f}")
        print(f"  ATLAS参照: {ATLAS_CONDITIONS[3]['Di_opt']:.3f} N at beta={ATLAS_CONDITIONS[3]['beta_opt']:.4f}")
    print()

    print("=" * 70)
    print("  Phase B 統合検証 完了")
    print("=" * 70)
