"""
structural_analyzer.py の検証スクリプト
=========================================
opt-ATLAS の構造計算（M, delta, sigma）と比較。

検証ポイント:
  1. 曲げモーメント M(y) の計算が opt-ATLAS と一致するか
  2. たわみ delta_tip の計算が opt-ATLAS と一致するか
  3. scipy 依存の問題（pip 遮断環境で動かない）
  4. check_strength の E=100GPa 固定値の問題
"""

import numpy as np
import sys
sys.path.insert(0, '.')

# ===========================================================================
# 0. scipy 依存チェック
# ===========================================================================
print("=== scipy 依存チェック ===")
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz_scipy
    SCIPY_AVAILABLE = True
    print("  scipy: 利用可能 (現在の環境では動く)")
except ImportError:
    try:
        from scipy.integrate import cumtrapz as cumtrapz_scipy
        SCIPY_AVAILABLE = True
        print("  scipy (旧API): 利用可能")
    except ImportError:
        SCIPY_AVAILABLE = False
        print("  scipy: 利用不可 ← pip 遮断環境では structural_analyzer.py は動かない!")
print()

# ===========================================================================
# 1. opt-ATLAS の構造計算を Python で再現（scipy 不使用）
# ===========================================================================

def cumtrapz_numpy(y, x, initial=0.0):
    """scipy の cumtrapz を純 NumPy で代替実装"""
    dx = np.diff(x)
    trapz = (y[:-1] + y[1:]) * dx / 2.0
    result = np.concatenate([[initial], np.cumsum(trapz)])
    return result


def opt_atlas_structural(y_m, Net_Load_Nm, EI_kgfmm2):
    """
    opt-ATLAS の構造計算ロジックを Python で再現。
    単位系: kgf・mm（opt-ATLAS と同じ）

    引数:
        y_m          : スパン方向座標 [m]
        Net_Load_Nm  : 正味荷重分布 [N/m]
        EI_kgfmm2    : 曲げ剛性分布 [kgf*mm^2]
    """
    g = 9.80665
    N = len(y_m)
    y_mm = y_m * 1000.0  # [mm]

    # --- 曲げモーメント（翼端から積分）---
    # opt-ATLAS と同じロジック
    M_Nm = np.zeros(N)
    for i in range(N - 1):
        dist = y_m[i:] - y_m[i]
        M_Nm[i] = np.trapezoid(Net_Load_Nm[i:] * dist, y_m[i:])
    M_Nm[N-1] = 0.0

    # 単位換算: N*m -> kgf*mm
    M_kgfmm = M_Nm * (1000.0 / g)  # 1 N*m = 1000/g kgf*mm

    # --- たわみ計算（opt-ATLAS: kgf*mm系）---
    curvature_per_mm = np.abs(M_kgfmm) / EI_kgfmm2          # [1/mm]
    theta_rad  = cumtrapz_numpy(curvature_per_mm, y_mm)      # [rad]
    delta_mm   = cumtrapz_numpy(theta_rad, y_mm)             # [mm]

    return M_Nm, M_kgfmm, delta_mm


def hemere_structural(y_m, Net_Load_Nm, EI_Nmm2):
    """
    Hemere の structural_analyzer と同じロジック（scipy を numpy で代替）。

    引数:
        y_m         : スパン方向座標 [m]
        Net_Load_Nm : 正味荷重分布 [N/m]
        EI_Nmm2     : 曲げ剛性分布 [N*mm^2]
    """
    N = len(y_m)

    # --- 曲げモーメント ---
    M_Nm = np.zeros(N)
    for i in range(N - 1):
        dist = y_m[i:] - y_m[i]
        M_Nm[i] = np.trapezoid(Net_Load_Nm[i:] * dist, y_m[i:])
    M_Nm[N-1] = 0.0

    # --- たわみ計算（Hemere: N*m^2 系）---
    EI_Nm2 = EI_Nmm2 * 1e-6        # N*mm^2 -> N*m^2
    curvature = np.abs(M_Nm) / (EI_Nm2 + 1e-9)   # [1/m]
    theta = cumtrapz_numpy(curvature, y_m)         # [rad]
    delta_m = cumtrapz_numpy(theta, y_m)           # [m]

    return M_Nm, delta_m * 1000.0  # delta を [mm] で返す


# ===========================================================================
# 2. 解析解との比較（等分布荷重・一様 EI のカンチレバー梁）
# ===========================================================================

def verify_with_analytic():
    """
    等分布荷重 w [N/m]、一様 EI のカンチレバー梁での解析解と比較。

    解析解:
      M(x) = w/2 * (L-x)^2
      delta_tip = w * L^4 / (8 * EI)
    """
    print("=== 解析解との比較（等分布荷重・一様EI カンチレバー梁）===")

    L = 16.0        # 半スパン [m]
    w = 100.0       # 等分布荷重 [N/m]
    EI_Nm2 = 1e6    # 曲げ剛性 [N*m^2]
    EI_Nmm2 = EI_Nm2 * 1e6   # [N*mm^2]
    EI_kgfmm2 = EI_Nmm2 / 9.80665  # [kgf*mm^2]
    N = 200

    y = np.linspace(0, L, N)
    load = np.full(N, w)

    # 解析解
    M_analytic = 0.5 * w * (L - y)**2
    delta_tip_analytic = w * L**4 / (8 * EI_Nm2) * 1000  # [mm]

    # opt-ATLAS 再現
    EI_kgfmm2_dist = np.full(N, EI_kgfmm2)
    M_Nm_atlas, M_kgfmm, delta_mm_atlas = opt_atlas_structural(y, load, EI_kgfmm2_dist)

    # Hemere 再現
    EI_Nmm2_dist = np.full(N, EI_Nmm2)
    M_Nm_hem, delta_mm_hem = hemere_structural(y, load, EI_Nmm2_dist)

    print(f"  解析解:      M(y=0) = {M_analytic[0]:.2f} N*m,  delta_tip = {delta_tip_analytic:.4f} mm")
    print(f"  opt-ATLAS:   M(y=0) = {M_Nm_atlas[0]:.2f} N*m,  delta_tip = {delta_mm_atlas[-1]:.4f} mm")
    print(f"  Hemere:      M(y=0) = {M_Nm_hem[0]:.2f} N*m,  delta_tip = {delta_mm_hem[-1]:.4f} mm")

    err_m_atlas = abs(M_Nm_atlas[0] - M_analytic[0]) / M_analytic[0] * 100
    err_d_atlas = abs(delta_mm_atlas[-1] - delta_tip_analytic) / delta_tip_analytic * 100
    err_m_hem   = abs(M_Nm_hem[0] - M_analytic[0]) / M_analytic[0] * 100
    err_d_hem   = abs(delta_mm_hem[-1] - delta_tip_analytic) / delta_tip_analytic * 100

    print(f"  opt-ATLAS 誤差: M={err_m_atlas:.3f}%, delta={err_d_atlas:.3f}%")
    print(f"  Hemere    誤差: M={err_m_hem:.3f}%, delta={err_d_hem:.3f}%")
    print()
    return delta_tip_analytic, delta_mm_atlas[-1], delta_mm_hem[-1]


# ===========================================================================
# 3. opt-ATLAS 条件3 で通し計算
# ===========================================================================

def verify_atlas_condition3_full():
    """
    opt-ATLAS 条件3 の全フロー（空力→構造）を再現して比較。

    条件3:
      W_total = 83.06 kg, span = 22m, V = 11m/s, rho = 1.154
      最適積層: [6,8,7,5,4,3,1,1]（全周積層）
      マンドレル径 [mm]: linspace(90,40,8) （直径として使う）
      beta = 0.9888
      期待値: delta_tip ≈ 1400mm（許容値）, sigma_margin > 0
    """
    print("=== opt-ATLAS 条件3 通し計算 ===")
    g = 9.80665

    # --- パラメータ ---
    W_total_kg = 83.06
    W_total_N  = W_total_kg * g
    span_m     = 22.0
    le         = span_m / 2.0
    V          = 11.0
    rho        = 1.154
    beta_opt   = 0.9888
    N_aero     = 100
    M_seg      = 8
    W_wing_fixed_kg = 8.0  # リブ・外皮等
    W_fixed_other_kg = 20 + 55  # 機体 + パイロット

    mandrel_D_mm = np.linspace(90, 40, M_seg)  # 直径 [mm]
    n_0deg_opt   = [6, 8, 7, 5, 4, 3, 1, 1]

    # --- 空力計算 ---
    from src.aerodynamics.aerodynamics_analyzer import AerodynamicsAnalyzer, AircraftAeroParams
    aero_params = AircraftAeroParams(
        lift_target_N=W_total_N,
        span_m=span_m,
        v_flight_ms=V,
        rho_air=rho,
        n_segments=N_aero,
    )
    aero = AerodynamicsAnalyzer(aero_params)
    aero_result = aero.solve(beta_opt)

    Di = aero_result["induced_drag_N"]
    L_dist = aero_result["lift_dist_N_m"]  # [N/m]
    y_aero = aero_result["span_y"]          # [m]

    print(f"  Di = {Di:.4f} N  (参考: opt-ATLAS 6.229N, 重量が違うため差あり)")

    # --- 構造計算用の EI 分布 ---
    from src.core.spar_calculator import SparCalculator
    calc = SparCalculator()

    y_struct = np.linspace(0, le, M_seg)  # [m]
    EI_kgfmm2_dist = np.zeros(M_seg)
    I_mm4_dist     = np.zeros(M_seg)
    D_outer_mm_dist= np.zeros(M_seg)
    w_struct_dist  = np.zeros(M_seg)  # [N/m]（構造自重）

    for j in range(M_seg):
        # 全周積層 + 集中積層なし（opt-ATLAS と同等）
        ply_counts = np.zeros(11)
        ply_counts[0]  = 1              # 24t 保護層
        ply_counts[1]  = 2              # 40t 45deg（トルク層 2枚）
        ply_counts[2]  = n_0deg_opt[j]  # 40t 0deg（主要曲げ層）← 全周扱い
        ply_counts[10] = 1              # 24t 保護層

        EI_kgf, w_kg_per_m, t_mm, _, _ = calc.calculate_spec(ply_counts, mandrel_D_mm[j])
        EI_kgfmm2_dist[j] = EI_kgf
        w_struct_dist[j]   = w_kg_per_m * g  # [N/m]

    EI_Nmm2_dist = EI_kgfmm2_dist * g  # kgf*mm^2 -> N*mm^2

    print(f"  EI 分布 [kgf*mm^2]: {EI_kgfmm2_dist}")
    print(f"  w_struct [N/m]: {np.round(w_struct_dist, 3)}")

    # --- 正味荷重 ---
    # y_aero グリッドに補間
    w_struct_aero = np.interp(y_aero, y_struct, w_struct_dist)
    w_fixed_per_m = (W_wing_fixed_kg * g) / span_m  # [N/m]
    Net_Load = L_dist - w_struct_aero - w_fixed_per_m

    # --- 構造計算（opt-ATLAS 再現）---
    EI_kgfmm2_aero = np.interp(y_aero, y_struct, EI_kgfmm2_dist)
    M_Nm, M_kgfmm, delta_mm = opt_atlas_structural(y_aero, Net_Load, EI_kgfmm2_aero)

    # --- 構造計算（Hemere 再現）---
    EI_Nmm2_aero = EI_kgfmm2_aero * g
    M_Nm_hem, delta_mm_hem = hemere_structural(y_aero, Net_Load, EI_Nmm2_aero)

    print(f"\n  --- 翼端たわみ ---")
    print(f"  opt-ATLAS 再現: delta_tip = {delta_mm[-1]:.1f} mm  (許容: 1400mm)")
    print(f"  Hemere    再現: delta_tip = {delta_mm_hem[-1]:.1f} mm")
    diff_d = abs(delta_mm[-1] - delta_mm_hem[-1]) / (abs(delta_mm[-1]) + 1e-6) * 100
    print(f"  両者の差: {diff_d:.3f}%")

    print(f"\n  --- 翼根曲げモーメント ---")
    print(f"  opt-ATLAS: M(y=0) = {M_Nm[0]:.1f} N*m = {M_kgfmm[0]:.1f} kgf*mm")
    print(f"  Hemere:    M(y=0) = {M_Nm_hem[0]:.1f} N*m")
    print()

    return delta_mm[-1], delta_mm_hem[-1]


# ===========================================================================
# 4. check_strength の E=100GPa 問題
# ===========================================================================

def check_strength_issue():
    """
    structural_analyzer.check_strength の E=100GPa 固定値の問題を検証。

    本来: σ = M * c / I
    Hemere: σ = M * c / (EI/E_fixed)  ← E を 100GPa 固定で I を逆算している

    正しいのは spar_calculator から I を直接取得すること。
    """
    print("=== check_strength の E固定問題 ===")

    # 例: 翼根 M=5000 N*m, EI=1e10 N*mm^2, D_outer=95mm
    M_Nm = 5000.0
    EI_Nmm2 = 1e10
    D_outer_mm = 95.0

    # Hemere の check_strength（E=100GPa 固定）
    E_fixed = 100e9   # 100 GPa = 100,000 N/mm^2 = 100 kN/mm^2
    I_m4_from_EI = (EI_Nmm2 * 1e-6) / E_fixed   # N*m^2 / Pa = m^4
    c_m = (D_outer_mm / 1000.0) / 2.0
    sigma_MPa_hemere = (M_Nm * c_m) / (I_m4_from_EI + 1e-12) / 1e6

    # 正しい計算: spar_calculator から I を直接取得
    # opt-ATLAS: Z = I / c  →  sigma = M / Z  （kgf*mm 系）
    # 今回は EI と D_outer から I を計算する方法の問題を示す
    from src.core.spar_calculator import SparCalculator
    calc = SparCalculator()
    ply = np.array([1, 2, 8, 1, 0, 0, 0, 0, 1, 1, 1], dtype=float)  # 典型的な翼根
    EI_kgf, w, t, _, _ = calc.calculate_spec(ply, 90.0)
    EI_Nmm2_real = EI_kgf * 9.80665

    # spar_calculator の内部計算で I を直接取得するには？
    # 現在の API では (EI, weight, thickness) しか返らない → I が取れない
    print("  問題1: structural_analyzer.check_strength は E=100GPa 固定で I を逆算している")
    print(f"    E_fixed = {E_fixed/1e9:.0f} GPa（等方性材料の典型値）")
    print(f"    実際の桁の等価ヤング率は積層構成次第で大きく異なる")
    print()
    print("  問題2: spar_calculator の API が (EI, weight, thickness) しか返さない")
    print(f"    -> I（断面二次モーメント）が取れないので正確な応力計算ができない")
    print()
    print("  正しい実装に必要なもの:")
    print("    - spar_calculator が I_mm4 と D_outer_mm も返すように修正")
    print("    - structural_analyzer.check_strength が I と D_outer を引数で受け取る")
    print()


# ===========================================================================
# メイン
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  structural_analyzer.py 検証")
    print("=" * 60)
    print()

    verify_with_analytic()
    verify_atlas_condition3_full()
    check_strength_issue()

    print("完了。")
