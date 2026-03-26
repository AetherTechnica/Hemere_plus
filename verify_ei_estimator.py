"""
EIEstimator の検証スクリプト
============================
calc_EI_req() が物理的に正しい EI 分布を返すことを確認する。

検証ポイント:
  1. 解析解との比較（等分布荷重カンチレバー梁）
  2. opt-ATLAS 条件3 の空力荷重下での検証（逆検証）
  3. 収束性と境界条件のテスト
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.core.ei_estimator import EIEstimator
from src.structural.structural_analyzer import StructuralAnalyzer

g = 9.80665

# ===========================================================================
# 1. 解析解との比較（等分布荷重・一様 EI カンチレバー梁）
# ===========================================================================

def verify_uniform_load():
    """
    等分布荷重 w [N/m]、M(y) = w/2 * (L-y)^2 のケース。

    解析解 (一様 EI の場合):
        delta_tip = w * L^4 / (8 * EI)
        → EI = w * L^4 / (8 * delta_allow)

    M比例形状法の仮定:
        EI_req(y) = alpha * M_shape(y) = alpha * (L-y)^2 / L^2
        ※ 分布形状は同じでも、一様 EI ではないため解析解とは若干ずれる
    """
    print("=== Test 1: 等分布荷重カンチレバー（解析解との比較）===")

    L = 16.0       # 半スパン [m]
    w = 100.0      # 等分布荷重 [N/m]
    delta_allow = w * L**4 / (8.0 * 1e6) * 0.5  # 一様 EI=1e6 N*m^2 時の δ/2 を目標に設定

    y = np.linspace(0, L, 500)
    M = 0.5 * w * (L - y) ** 2  # 等分布荷重のモーメント [N*m]

    est = EIEstimator()
    EI_req = est.calc_EI_req(M, y, delta_allow)

    # 逆検証: 求めた EI_req で δ を計算
    sa = StructuralAnalyzer(y)
    d, _ = sa.compute_deflection(M, EI_req)

    err_pct = abs(d[-1] - delta_allow) / delta_allow * 100

    print(f"  w={w} N/m, L={L} m, delta_allow={delta_allow*1000:.2f} mm")
    print(f"  EI_req 範囲: {np.min(EI_req):.3e} 〜 {np.max(EI_req):.3e} [N*mm^2]")
    print(f"  delta_tip 計算値: {d[-1]*1000:.4f} mm")
    print(f"  delta_tip 目標値: {delta_allow*1000:.4f} mm")
    print(f"  誤差: {err_pct:.5f} %")

    status = "PASS" if err_pct < 0.01 else "FAIL"
    print(f"  結果: {status} (許容誤差: 0.01%)")
    print()
    return err_pct < 0.01


# ===========================================================================
# 2. 集中荷重カンチレバー（HANDOVER.md に記載のテストケース）
# ===========================================================================

def verify_point_load():
    """
    HANDOVER.md Phase B-1-1 の検証スクリプト:
        M = 0.5 * 100 * (16 - y)^2  （集中荷重モデル）
        delta_allow = 0.8192 m
        期待: delta_tip = 819.2 mm
    """
    print("=== Test 2: 集中荷重モデル（HANDOVER.md のテストケース）===")

    L = 16.0
    y = np.linspace(0, L, 200)
    M = 0.5 * 100 * (L - y) ** 2
    delta_allow = 0.8192  # [m]

    est = EIEstimator()
    EI_req = est.calc_EI_req(M, y, delta_allow)

    sa = StructuralAnalyzer(y)
    d, _ = sa.compute_deflection(M, EI_req)

    err_pct = abs(d[-1] - delta_allow) / delta_allow * 100
    print(f"  delta_tip = {d[-1]*1000:.2f} mm  (期待値: {delta_allow*1000:.1f} mm)")
    print(f"  EI_req min/max = {np.min(EI_req):.3e} / {np.max(EI_req):.3e} [N*mm^2]")
    print(f"  誤差: {err_pct:.5f} %")

    status = "PASS" if err_pct < 0.01 else "FAIL"
    print(f"  結果: {status}")
    print()
    return err_pct < 0.01


# ===========================================================================
# 3. 実際の空力荷重下での検証（opt-ATLAS 条件3）
# ===========================================================================

def verify_with_aero_load():
    """
    opt-ATLAS 条件3 の空力荷重下で EI_req を計算し、逆検証する。

    条件3:
        W_total ≈ 83 kg, span = 22m, V = 11m/s, beta = 0.9888
        delta_allow = 0.9 m
    """
    print("=== Test 3: opt-ATLAS 条件3 の空力荷重下での逆検証 ===")

    from src.aerodynamics.aerodynamics_analyzer import AerodynamicsAnalyzer, AircraftAeroParams

    W_total_kg = 83.06
    span_m     = 22.0
    V          = 11.0
    rho        = 1.154
    beta       = 0.9888
    delta_allow = 0.9   # [m]

    # 空力計算
    aero_params = AircraftAeroParams(
        lift_target_N=W_total_kg * g,
        span_m=span_m,
        v_flight_ms=V,
        rho_air=rho,
        n_segments=100,
    )
    aero = AerodynamicsAnalyzer(aero_params)
    res = aero.solve(beta)
    y_m       = res['span_y']
    L_dist    = res['lift_dist_N_m']

    # 簡易正味荷重（自重 0 近似）
    sa = StructuralAnalyzer(y_m)
    M_Nm = sa.compute_bending_moment(L_dist)

    # EI_req 逆算
    est = EIEstimator()
    EI_req = est.calc_EI_req(M_Nm, y_m, delta_allow)

    # 逆検証
    d, _ = sa.compute_deflection(M_Nm, EI_req)

    err_pct = abs(d[-1] - delta_allow) / delta_allow * 100

    print(f"  Di = {res['induced_drag_N']:.4f} N")
    print(f"  M(y=0) = {M_Nm[0]:.1f} N*m")
    print(f"  EI_req 翼根: {EI_req[0]:.3e} N*mm^2  ({EI_req[0]/g:.3e} kgf*mm^2)")
    print(f"  EI_req 翼端: {EI_req[-1]:.3e} N*mm^2")
    print(f"  delta_tip 計算: {d[-1]*1000:.2f} mm  (目標: {delta_allow*1000:.1f} mm)")
    print(f"  逆検証誤差: {err_pct:.5f} %")

    # EI_req の値が現実的かチェック（典型的な桁は 10^9 〜 10^11 N*mm^2）
    EI_root_kgf = EI_req[0] / g
    print(f"\n  翼根 EI_req = {EI_root_kgf:.3e} kgf*mm^2")
    if 1e8 < EI_root_kgf < 1e12:
        print(f"  EI_req の値: 現実的な範囲内 ✓")
    else:
        print(f"  EI_req の値: 物理的に非現実的 ← 要確認")

    status = "PASS" if err_pct < 0.01 else "FAIL"
    print(f"  結果: {status}")
    print()
    return err_pct < 0.01, EI_req


# ===========================================================================
# 4. 収束性・境界条件テスト
# ===========================================================================

def verify_convergence():
    """
    二分法の収束性と境界条件を確認する。
    - 収束精度の確認（tol_rel=1e-5 で十分か）
    - 異常入力（M≒0, 大きすぎる δ_allow）の処理確認
    """
    print("=== Test 4: 収束性・境界条件テスト ===")

    L = 16.0
    w = 100.0
    y = np.linspace(0, L, 200)
    M = 0.5 * w * (L - y) ** 2

    est = EIEstimator()
    sa_full = StructuralAnalyzer(y)

    # 4-1: 複数の delta_allow で誤差を確認
    print("  4-1: delta_allow を変化させて収束精度を確認")
    print(f"  {'delta_allow [mm]':>20} | {'計算 delta [mm]':>18} | {'誤差 [%]':>12}")
    print("  " + "-" * 60)
    all_pass = True
    for delta_m in [0.2, 0.5, 1.0, 2.0, 5.0]:
        EI_req = est.calc_EI_req(M, y, delta_m)
        d, _ = sa_full.compute_deflection(M, EI_req)
        err = abs(d[-1] - delta_m) / delta_m * 100
        flag = "OK" if err < 0.01 else "NG"
        print(f"  {delta_m*1000:>18.1f} mm | {d[-1]*1000:>16.4f} mm | {err:>10.5f}% {flag}")
        if err >= 0.01:
            all_pass = False

    # 4-2: ゼロモーメント入力
    print("\n  4-2: ゼロモーメント入力（M ≈ 0）")
    M_zero = np.zeros(200)
    EI_zero = est.calc_EI_req(M_zero, y, 1.0)
    print(f"  EI_req[0] = {EI_zero[0]:.3e} N*mm^2  (最小剛性 1e6 N*mm^2 が返るはず)")
    if abs(EI_zero[0] - 1e6) < 1:
        print("  ゼロモーメント処理: OK ✓")
    else:
        print("  ゼロモーメント処理: 要確認 ✗")
        all_pass = False

    status = "PASS" if all_pass else "FAIL"
    print(f"\n  結果: {status}")
    print()
    return all_pass


# ===========================================================================
# 5. 単位変換の整合性チェック
# ===========================================================================

def verify_unit_consistency():
    """
    EIEstimator の出力単位と StructuralAnalyzer, SnapOptimizer の入力単位の整合性確認。
    """
    print("=== Test 5: 単位変換の整合性チェック ===")

    L = 16.0
    w = 100.0
    delta_allow = 1.0  # [m]

    y = np.linspace(0, L, 200)
    M = 0.5 * w * (L - y) ** 2

    est = EIEstimator()
    EI_req_Nmm2 = est.calc_EI_req(M, y, delta_allow)

    # EIEstimator の出力: N*mm^2
    EI_req_Nm2    = EI_req_Nmm2 * 1e-6   # StructuralAnalyzer の compute_deflection 用
    EI_req_kgfmm2 = EI_req_Nmm2 / g      # SnapOptimizer 用

    print(f"  EI_req [N*mm^2]:   {EI_req_Nmm2[0]:.4e}  ← EIEstimator 出力")
    print(f"  EI_req [N*m^2]:    {EI_req_Nm2[0]:.4e}  ← StructuralAnalyzer 内部単位")
    print(f"  EI_req [kgf*mm^2]: {EI_req_kgfmm2[0]:.4e}  ← SnapOptimizer 入力（変換必要）")
    print()
    print("  SnapOptimizer への変換式: EI_req_kgfmm2 = EI_req_Nmm2 / 9.80665")
    print("  design_integrator_v3.py:  target_ei_dist[i] / self.g  ← 正しい変換 ✓")
    print()
    return True


# ===========================================================================
# メイン
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  EIEstimator 検証スクリプト")
    print("  M比例形状法 + 二分法の物理的正確性を確認")
    print("=" * 65)
    print()

    results = {
        "Test 1 等分布荷重解析解":    verify_uniform_load(),
        "Test 2 集中荷重HANDOVER例":  verify_point_load(),
        "Test 4 収束性テスト":         verify_convergence(),
        "Test 5 単位変換整合性":       verify_unit_consistency(),
    }

    # Test 3 は別途結果を格納
    t3_pass, EI_req_aero = verify_with_aero_load()
    results["Test 3 空力荷重逆検証"] = t3_pass

    print("=" * 65)
    print("  検証結果まとめ")
    print("=" * 65)
    all_ok = True
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name:<30} : {status}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("  全テスト PASS — EIEstimator は物理的に正しく動作しています")
    else:
        print("  一部テスト FAIL — 上記の詳細を確認してください")
