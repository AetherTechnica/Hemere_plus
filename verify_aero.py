"""
aerodynamics_analyzer.py の検証スクリプト
==========================================
opt-ATLAS の calculateAerodynamics と同じ入力を与えて Di が一致するか確認。

検証ポイント:
  1. 行列 A の生成方向（列 vs 行ブロードキャスト）
  2. 連立方程式の構造
  3. Di, 循環分布の数値的一致
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from src.aerodynamics.aerodynamics_analyzer import AerodynamicsAnalyzer, AircraftAeroParams


# ===========================================================================
# 1. opt-ATLAS の calculateAerodynamics を Python で完全再現
# ===========================================================================

def opt_atlas_calculate_aerodynamics(W_total_N, span_m, V_flight, rho_air, N, beta):
    """
    opt-ATLAS の calculateAerodynamics + createAircraft を Python で再現。
    MATLABコードのロジックをそのまま移植。
    """
    le = span_m / 2.0
    delta_S = np.ones(N) * (le / N / 2.0)
    y = np.linspace(delta_S[0], le - delta_S[-1], N)
    z = np.zeros(N)
    phi = np.zeros(N)

    # --- Q行列の計算（opt-ATLAS の calc_param + calc_Q に対応）---
    y_col = y[:, np.newaxis]
    z_col = z[:, np.newaxis]
    phi_col = phi[:, np.newaxis]

    ydash  = (y_col - y) * np.cos(phi) + (z_col - z) * np.sin(phi)
    zdash  = -(y_col - y) * np.sin(phi) + (z_col - z) * np.cos(phi)
    y2dash = (y_col + y) * np.cos(phi) + (z_col - z) * np.sin(phi)
    z2dash = -(y_col + y) * np.sin(phi) + (z_col - z) * np.cos(phi)

    R_plus     = (ydash  - delta_S)**2 + zdash**2
    R_minus    = (ydash  + delta_S)**2 + zdash**2
    Rdash_plus = (y2dash + delta_S)**2 + z2dash**2
    Rdash_minus= (y2dash - delta_S)**2 + z2dash**2

    term1 = (-(ydash - delta_S) / R_plus  + (ydash + delta_S) / R_minus ) * np.cos(phi_col - phi)
    term2 = (-zdash / R_plus  + zdash / R_minus ) * np.sin(phi_col - phi)
    term3 = (-(y2dash - delta_S) / Rdash_minus + (y2dash + delta_S) / Rdash_plus) * np.cos(phi_col + phi)
    term4 = (-z2dash / Rdash_minus + z2dash / Rdash_plus) * np.sin(phi_col + phi)
    Q = (1.0 / (2.0 * np.pi)) * (term1 + term2 + term3 + term4)

    # --- 正規化（opt-ATLAS の normalize に対応）---
    delta_sigma = delta_S / le
    eta = y / le
    q_mat = Q * le  # = Q * le

    # --- 行列A の生成 ---
    # opt-ATLAS: A = pi * q_mat .* delta_sigma'
    # delta_sigma' は行ベクトル → 列ごとにスカラーを掛ける（列ブロードキャスト）
    # A_ij = pi * q_ij * delta_sigma_j
    A_atlas = np.pi * q_mat * delta_sigma[np.newaxis, :]  # 列ブロードキャスト

    # --- 連立方程式の構築 ---
    c_vec = (2.0 * delta_sigma)
    b_vec = (3.0 * np.pi / 2.0) * delta_sigma * eta

    sys_mat = np.zeros((N + 2, N + 2))
    sys_mat[:N, :N] = A_atlas + A_atlas.T
    sys_mat[:N,  N]  = -c_vec
    sys_mat[:N,  N+1]= -b_vec
    sys_mat[N,  :N]  = -c_vec
    sys_mat[N+1,:N]  = -b_vec

    rhs = np.zeros(N + 2)
    rhs[N]   = -1.0
    rhs[N+1] = -beta

    # 求解
    rcond_val = np.linalg.cond(sys_mat)
    if rcond_val > 1e16:
        return None

    sol = np.linalg.solve(sys_mat, rhs)
    g = sol[:N]

    # --- 物理量の復元 ---
    Gamma = (W_total_N / (2.0 * le * rho_air * V_flight)) * g
    L_dist = rho_air * V_flight * Gamma  # [N/m]

    Di_ellipse = W_total_N**2 / (2.0 * np.pi * rho_air * V_flight**2 * le**2)
    efficiency_inv = g.T @ A_atlas @ g
    Di = efficiency_inv * Di_ellipse

    return {
        "Di": Di,
        "gamma": Gamma,
        "L_dist": L_dist,
        "y": y,
        "efficiency": 1.0 / efficiency_inv,
        "A_atlas": A_atlas,
        "g": g,
    }


# ===========================================================================
# 2. 行列 A のブロードキャスト方向の確認
# ===========================================================================

def check_A_matrix_direction():
    """
    opt-ATLAS:  A = pi * Q * delta_sigma'（列ブロードキャスト: A_ij = pi*q_ij*ds_j）
    Hemere:     A = pi * Q * delta_sigma[:, newaxis]（行ブロードキャスト: A_ij = pi*q_ij*ds_i）

    等間隔の場合 ds_i = ds_j なので結果は同じ。
    非等間隔では違う。
    """
    print("=== A 行列のブロードキャスト方向確認 ===")

    N = 5
    q_test = np.random.rand(N, N)
    ds = np.array([0.1, 0.2, 0.15, 0.12, 0.18])  # 不等間隔でテスト

    A_col = np.pi * q_test * ds[np.newaxis, :]  # opt-ATLAS（列ブロードキャスト）
    A_row = np.pi * q_test * ds[:, np.newaxis]  # Hemere（行ブロードキャスト）

    diff_max = np.max(np.abs(A_col - A_row))
    print(f"  不等間隔(ds={ds})での A_col vs A_row の最大差: {diff_max:.6f}")
    print(f"  -> 不等間隔では {'一致しない' if diff_max > 1e-10 else '一致する'}")

    # 等間隔（Hemereの実際の使用ケース）
    ds_eq = np.ones(N) * 0.1
    A_col_eq = np.pi * q_test * ds_eq[np.newaxis, :]
    A_row_eq = np.pi * q_test * ds_eq[:, np.newaxis]
    diff_eq = np.max(np.abs(A_col_eq - A_row_eq))
    print(f"  等間隔(ds=0.1)での A_col vs A_row の最大差: {diff_eq:.2e}")
    print(f"  -> 等間隔では {'一致する' if diff_eq < 1e-10 else '一致しない'}")
    print()


# ===========================================================================
# 3. opt-ATLAS 条件別の Di 比較
# ===========================================================================

def compare_conditions():
    """
    opt-ATLAS の各条件での Di を、Hemere の aerodynamics_analyzer と比較。
    """
    g = 9.80665

    conditions = [
        {
            "name": "条件1（低速ロングスパン）",
            "W_other_kg": 33 + 58,
            "W_struct_kg": 25.15,  # opt-ATLASの結果
            "span_m": 32.5,
            "V": 7.2,
            "beta": 0.9794,
            "Di_expected": 10.301,
        },
        {
            "name": "条件3（高速ショートスパン）",
            "W_other_kg": 20 + 55 + 8,  # W_fixed_other + W_wing_fixed
            "W_struct_kg": 13.06,  # opt-ATLASの最適解での主翼重量（桁のみ）
            "span_m": 22.0,
            "V": 11.0,
            "beta": 0.9888,
            "Di_expected": 6.229,
        },
    ]

    print("=== opt-ATLAS 各条件の Di 比較 ===")
    for cond in conditions:
        W_total_N = (cond["W_other_kg"] + cond["W_struct_kg"]) * g

        # opt-ATLAS 再現
        res_atlas = opt_atlas_calculate_aerodynamics(
            W_total_N, cond["span_m"], cond["V"], 1.154, 100, cond["beta"]
        )

        # Hemere
        params = AircraftAeroParams(
            lift_target_N=W_total_N,
            span_m=cond["span_m"],
            v_flight_ms=cond["V"],
            rho_air=1.154,
            n_segments=100,
        )
        analyzer = AerodynamicsAnalyzer(params)
        res_hemere = analyzer.solve(cond["beta"])

        print(f"\n--- {cond['name']} ---")
        print(f"  W_total = {W_total_N / g:.2f} kg = {W_total_N:.1f} N")
        print(f"  beta = {cond['beta']}")
        print(f"  Di 期待値（opt-ATLAS MATLAB）: {cond['Di_expected']:.4f} N")
        if res_atlas:
            print(f"  Di opt-ATLAS 再現（Python）:   {res_atlas['Di']:.4f} N")
            diff_atlas = abs(res_atlas['Di'] - cond['Di_expected']) / cond['Di_expected'] * 100
            print(f"    MATLAB との差: {diff_atlas:.3f}%")
        if res_hemere:
            print(f"  Di Hemere:                     {res_hemere['induced_drag_N']:.4f} N")
            if res_atlas:
                diff_hem = abs(res_hemere['induced_drag_N'] - res_atlas['Di']) / res_atlas['Di'] * 100
                print(f"    opt-ATLAS 再現との差: {diff_hem:.3f}%")
    print()


# ===========================================================================
# 4. beta スイープ（Di vs beta の形状確認）
# ===========================================================================

def beta_sweep():
    """
    beta を変えたとき Di がどう変わるかを確認。
    opt-ATLAS の実績: beta < 1.0 で Di が最小になるはず。
    """
    g = 9.80665
    W_total_N = 83.06 * g  # 条件3の最適解での総重量
    span_m = 22.0
    V = 11.0
    rho = 1.154
    N = 100

    betas = np.linspace(0.85, 1.05, 21)

    print("=== beta スイープ（条件3）===")
    print(f"  {'beta':>6} | {'Di_atlas[N]':>12} | {'Di_hemere[N]':>13} | {'diff%':>7}")
    print("  " + "-" * 50)

    Di_atlas_list = []
    Di_hemere_list = []

    params = AircraftAeroParams(
        lift_target_N=W_total_N,
        span_m=span_m,
        v_flight_ms=V,
        rho_air=rho,
        n_segments=N,
    )
    analyzer = AerodynamicsAnalyzer(params)

    for beta in betas:
        res_a = opt_atlas_calculate_aerodynamics(W_total_N, span_m, V, rho, N, beta)
        res_h = analyzer.solve(beta)

        di_a = res_a["Di"] if res_a else float('nan')
        di_h = res_h["induced_drag_N"] if res_h else float('nan')
        Di_atlas_list.append(di_a)
        Di_hemere_list.append(di_h)

        diff = abs(di_h - di_a) / di_a * 100 if (not np.isnan(di_a) and di_a > 0) else float('nan')
        print(f"  {beta:>6.3f} | {di_a:>12.4f} | {di_h:>13.4f} | {diff:>7.3f}%")

    # 最小 Di の beta を確認
    Di_arr = np.array(Di_atlas_list)
    valid = ~np.isnan(Di_arr)
    if valid.any():
        best_idx = np.argmin(Di_arr[valid])
        betas_valid = betas[valid]
        print(f"\n  opt-ATLAS 再現での最適 beta: {betas_valid[best_idx]:.3f}  (Di={Di_arr[valid][best_idx]:.4f} N)")
    print()


# ===========================================================================
# メイン
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  aerodynamics_analyzer.py 検証")
    print("=" * 60)
    print()

    check_A_matrix_direction()
    compare_conditions()
    beta_sweep()

    print("完了。")
