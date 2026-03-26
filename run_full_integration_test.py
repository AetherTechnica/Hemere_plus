"""
opt-ATLAS 3条件での統合検証スクリプト
======================================
Hemere の計算結果を opt-ATLAS の参照値と比較して精度を確認する。

検証条件:
  条件1: スパン 32.5m, 速度 7.2m/s  -> beta ≈ 0.9794, Di ≈ 10.301N
  条件2: スパン 29m,   速度 8.3m/s  -> beta ≈ 0.9809, Di ≈ 7.670N
  条件3: スパン 22m,   速度 11m/s   -> beta ≈ 0.9888, Di ≈ 6.229N

比較方法:
  - beta スイープ最適化を実施し、Di が最小となる beta* を探索
  - Di は W_total（W_fixed + W_spar）に依存するため、傾向確認が主目的
  - W_spar は LayupOptimizer の最適化結果として得られる
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.aerodynamics.aerodynamics_analyzer import AerodynamicsAnalyzer, AircraftAeroParams
from src.integration.design_integrator_v3 import DesignIntegratorV3, WingDesignParams

g = 9.80665

# ===========================================================================
# opt-ATLAS 参照値
# ===========================================================================
ATLAS_CONDITIONS = {
    1: {
        "name": "条件1 (32.5m, 7.2m/s)",
        "W_pilot_kg": 58.0, "W_fuselage_kg": 19.0, "W_wing_secondary_kg": 9.0,
        "span_m": 32.5, "V": 7.2, "delta_allow_m": 2.2,
        "sigma_allow_MPa": 300.0,
        "mandrel_D": np.linspace(120, 45, 8),
        "beta_opt": 0.9794, "Di_opt": 10.301,
    },
    2: {
        "name": "条件2 (29m, 8.3m/s)",
        "W_pilot_kg": 58.0, "W_fuselage_kg": 19.0, "W_wing_secondary_kg": 9.0,
        "span_m": 29.0, "V": 8.3, "delta_allow_m": 2.0,
        "sigma_allow_MPa": 300.0,
        "mandrel_D": np.linspace(110, 40, 8),
        "beta_opt": 0.9809, "Di_opt": 7.670,
    },
    3: {
        "name": "条件3 (22m, 11m/s)",
        "W_pilot_kg": 55.0, "W_fuselage_kg": 20.0, "W_wing_secondary_kg": 8.0,
        "span_m": 22.0, "V": 11.0, "delta_allow_m": 0.9,
        "sigma_allow_MPa": 300.0,
        "mandrel_D": np.linspace(90, 40, 8),
        "beta_opt": 0.9888, "Di_opt": 6.229,
    },
}


# ===========================================================================
# 1. 空力計算の検証（beta 固定で Di を計算）
# ===========================================================================

def verify_aero_only(cond_id):
    """
    opt-ATLAS の beta_opt を固定して Di を計算し、空力計算精度を確認する。
    W_total = W_fixed（W_spar=0 近似）で計算。
    """
    cond = ATLAS_CONDITIONS[cond_id]
    W_total_kg = cond['W_pilot_kg'] + cond['W_fuselage_kg'] + cond['W_wing_secondary_kg']

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
# 2. 統合フロー（空力 + 構造 + 積層最適化 + beta スイープ）
# ===========================================================================

def run_integration_sweep(cond_id, beta_min=0.97, beta_max=1.01, n_beta=9, verbose=True):
    """
    指定条件で Hemere の統合フロー（beta スイープ）を実行する。
    W_total = W_fixed + W_spar（W_spar は LayupOptimizer の最適化結果）。
    """
    cond = ATLAS_CONDITIONS[cond_id]

    params = WingDesignParams(
        W_pilot_kg          = cond['W_pilot_kg'],
        W_fuselage_kg       = cond['W_fuselage_kg'],
        W_wing_secondary_kg = cond['W_wing_secondary_kg'],
        span_m              = cond['span_m'],
        V_flight_ms         = cond['V'],
        delta_allow_m       = cond['delta_allow_m'],
        sigma_allow_MPa     = cond['sigma_allow_MPa'],
        mandrel_diameters_mm= cond['mandrel_D'],
        n_segments          = 8,
    )

    integrator = DesignIntegratorV3(params)
    result = integrator.optimize(
        beta_min=beta_min,
        beta_max=beta_max,
        n_beta=n_beta,
        verbose=verbose,
    )
    return result


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
    # Step A: 空力計算の精度確認（W_spar=0 近似、beta 固定）
    # -----------------------------------------------------------------
    print("-" * 70)
    print("Step A: 空力計算精度（beta=opt-ATLAS最適値 固定、W_spar=0近似）")
    print("-" * 70)
    print(f"{'条件':^18} | {'Hemere Di [N]':>14} | {'ATLAS Di [N]':>13} | {'誤差 [%]':>10}")
    print("-" * 70)
    for cond_id in [1, 2, 3]:
        Di, err = verify_aero_only(cond_id)
        name = ATLAS_CONDITIONS[cond_id]['name']
        Di_ref = ATLAS_CONDITIONS[cond_id]['Di_opt']
        flag = "OK" if abs(err) < 5.0 else "NG!"
        print(f" {name:<18} | {Di:>13.4f} N | {Di_ref:>12.3f} N | {err:>+9.2f}% {flag}")
    print()
    print("  注: W_total の差（桁自重の扱い）により Di に差が出る。")
    print("  空力計算自体は verify_aero.py で 0.000% 精度検証済み。")
    print()

    # -----------------------------------------------------------------
    # Step B: 統合フロー（beta スイープ + W_total 収束）
    # -----------------------------------------------------------------
    print("-" * 70)
    print("Step B: 統合フロー（W_total収束 + beta スイープ最適化）")
    print("-" * 70)

    for cond_id in [3, 2, 1]:
        cond = ATLAS_CONDITIONS[cond_id]
        print(f"\n{cond['name']}:")
        result = run_integration_sweep(cond_id, verbose=True)

        if result:
            print()
            print(f"  [opt-ATLAS 参照]")
            print(f"  beta_opt = {cond['beta_opt']:.4f},  Di_opt = {cond['Di_opt']:.3f} N")
            print(f"  [Hemere 最適解]")
            print(f"  beta*    = {result['beta']:.4f},  Di*    = {result['induced_drag_N']:.4f} N")
            delta_err = (result['induced_drag_N'] - cond['Di_opt']) / cond['Di_opt'] * 100
            print(f"  Di 誤差  = {delta_err:+.2f}%")
            print(f"  W_spar   = {result['W_spar_kg']:.3f} kg")
            print(f"  W_total  = {result['W_total_kg']:.3f} kg  (W_fixed={cond['W_pilot_kg']+cond['W_fuselage_kg']+cond['W_wing_secondary_kg']:.1f}kg + W_spar)")
        print()

    print("=" * 70)
    print("  Phase B 統合検証 完了")
    print("=" * 70)
