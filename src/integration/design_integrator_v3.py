import numpy as np
import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# =========================================================
# Pickleロード用定義
# =========================================================
def add_physics_features(X):
    log_ei = X[:, 0]
    r = X[:, 1]
    log_r = np.log10(r + 1e-9)
    thickness_index = log_ei - 3 * log_r
    weight_index    = log_ei - 2 * log_r
    return np.column_stack((X, thickness_index, weight_index))

class PhysicsFeatureEngineer:
    def transform(self, X):
        return add_physics_features(X)

def inverse_log10(x):
    return 10**x
# =========================================================

from src.aerodynamics.aerodynamics_analyzer import AerodynamicsAnalyzer, AircraftAeroParams
from src.structural.structural_analyzer import StructuralAnalyzer
from src.core.ei_estimator import EIEstimator
from src.core.layup_optimizer import LayupOptimizer

class DesignIntegratorV3:
    def __init__(self, aero_params: AircraftAeroParams,
                 mandrel_diameters_mm: np.ndarray = None):
        """
        Args:
            aero_params          : 空力計算パラメータ
            mandrel_diameters_mm : マンドレル直径分布 [mm]（n_segments 個）
                                   None の場合は linspace(90, 40, n_segments) を使用
        """
        self.aero = AerodynamicsAnalyzer(aero_params)
        self.struct = StructuralAnalyzer(self.aero.y)
        self.ei_est = EIEstimator()
        self.layup_opt = LayupOptimizer()
        self.g = 9.80665
        n = aero_params.n_segments
        if mandrel_diameters_mm is None:
            self.mandrel_D = np.linspace(90.0, 40.0, n)
        else:
            assert len(mandrel_diameters_mm) == n, \
                f"mandrel_diameters_mm の長さ({len(mandrel_diameters_mm)})が n_segments({n})と不一致"
            self.mandrel_D = np.array(mandrel_diameters_mm, dtype=float)

    def optimize_full_wing(self, beta=0.9, max_deflection_m=1.5, verbose=True,
                           max_weight_iter=5, weight_tol=0.01):
        """
        空力→EI逆算→積層最適化→最終検証の統合フロー（重量収束ループ付き）。

        Args:
            beta             : 翼根モーメント低減率 (0.8 〜 1.0)
            max_deflection_m : 許容翼端たわみ [m]
            verbose          : 途中経過の出力フラグ
            max_weight_iter  : 重量収束ループの最大反復回数
            weight_tol       : 重量収束判定の相対許容誤差

        Returns:
            dict: {
                'beta': float,
                'induced_drag_N': float,
                'W_struct_kg': float,
                'delta_tip_m': float,
                'sigma_max_MPa': float,
                'wing_specs': list[dict],  各セグメントの積層情報
                'feasible': bool,
            }
        """
        dy = None  # スパン刻み（初回計算後に設定）

        # 初期自重ゼロ近似からスタート
        w_struct_prev = np.zeros(len(self.aero.y))  # [N/m]

        for weight_iter in range(max_weight_iter):
            # --- Step 1: 空力解析（W_total = 基本重量）---
            aero_res = self.aero.solve(beta=beta)
            if not aero_res:
                return None

            if dy is None:
                dy = self.aero.y[1] - self.aero.y[0] if len(self.aero.y) > 1 else 1.0

            # --- Step 2: 必要剛性の逆算（EIEstimator）---
            # 正味荷重（揚力 - 桁自重）で M を計算
            net_load_for_EI = aero_res['lift_dist_N_m'] - w_struct_prev
            M_Nm = self.struct.compute_bending_moment(net_load_for_EI)
            EI_req_Nmm2 = self.ei_est.calc_EI_req(M_Nm, self.aero.y, max_deflection_m)
            EI_req_kgf  = EI_req_Nmm2 / self.g

            # --- Step 3: 積層最適化（LayupOptimizer）---
            wing_specs = []
            infeasible_count = 0
            for i, (ei_req, D) in enumerate(zip(EI_req_kgf, self.mandrel_D)):
                res = self.layup_opt.find_min_weight_layup(ei_req, D)
                wing_specs.append(res)
                if not res['feasible']:
                    infeasible_count += 1

            # 新しい自重分布を計算
            actual_weight = np.array([s['weight'] for s in wing_specs])  # [kg/m]
            w_struct_new  = actual_weight * self.g  # [N/m]

            # 収束チェック（前回との相対変化）
            diff = np.max(np.abs(w_struct_new - w_struct_prev)) / (np.max(w_struct_new) + 1e-9)
            if verbose and max_weight_iter > 1:
                W_kg = np.sum(actual_weight) * dy * 2.0
                print(f" [iter {weight_iter+1}] W_struct={W_kg:.3f} kg, 収束誤差={diff:.4f}")

            w_struct_prev = w_struct_new
            if diff < weight_tol:
                break  # 収束

        if verbose:
            print(f"\n{'='*60}")
            print(f" Hemere V3: beta={beta:.4f}, delta_allow={max_deflection_m:.2f} m")
            print(f"{'='*60}")
            for i, s in enumerate(wing_specs):
                if not s['feasible']:
                    print(f" [WARN] seg {i+1}: infeasible ({s['reason']})")

        # --- Step 4: 最終構造解析（答え合わせ）---
        actual_EI_kgf   = np.array([s['EI_actual']  for s in wing_specs])
        actual_weight   = np.array([s['weight']      for s in wing_specs])
        actual_I_mm4    = np.array([s['I_mm4']       for s in wing_specs])
        actual_D_outer  = np.array([s['D_outer_mm']  for s in wing_specs])

        # 正味荷重（揚力 - 桁自重）を再計算
        w_struct_N_m = actual_weight * self.g
        net_load = aero_res['lift_dist_N_m'] - w_struct_N_m

        # 構造解析: EI [kgf*mm^2] → [N*mm^2] 変換
        final_M     = self.struct.compute_bending_moment(net_load)
        final_d, _  = self.struct.compute_deflection(final_M, actual_EI_kgf * self.g)
        final_sigma = self.struct.check_strength(final_M, actual_I_mm4, actual_D_outer)

        # 主翼重量（半スパン積分 × 2）
        dy = self.aero.y[1] - self.aero.y[0] if len(self.aero.y) > 1 else 1.0
        W_struct_kg = np.sum(actual_weight) * dy * 2.0

        delta_tip    = float(final_d[-1])
        sigma_max    = float(np.max(final_sigma))
        feasible_all = (infeasible_count == 0) and (delta_tip <= max_deflection_m * 1.05)

        if verbose:
            print(f"\n{'='*60}")
            print(f" FINAL VERIFICATION")
            print(f"{'='*60}")
            print(f" beta              : {beta:.4f}")
            print(f" Di (誘導抗力)     : {aero_res['induced_drag_N']:.4f} N")
            print(f" Target Deflection : {max_deflection_m*1000:.1f} mm")
            print(f" Actual Deflection : {delta_tip*1000:.1f} mm")
            print(f" Max Fiber Stress  : {sigma_max:.1f} MPa")
            print(f" W_struct (桁)     : {W_struct_kg:.3f} kg")
            print(f" Infeasible segs   : {infeasible_count} / {len(wing_specs)}")
            print(f"{'='*60}")
            if delta_tip > max_deflection_m * 1.05:
                print(" [WARNING] Deflection exceeds target by >5%!")

        return {
            'beta'          : beta,
            'induced_drag_N': float(aero_res['induced_drag_N']),
            'W_struct_kg'   : W_struct_kg,
            'delta_tip_m'   : delta_tip,
            'sigma_max_MPa' : sigma_max,
            'wing_specs'    : wing_specs,
            'feasible'      : feasible_all,
        }

if __name__ == "__main__":
    g = 9.80665
    params = AircraftAeroParams(
        lift_target_N=83.06 * g,
        span_m=22.0,
        v_flight_ms=11.0,
        rho_air=1.154,
        n_segments=8
    )
    mandrel_D = np.linspace(90, 40, 8)
    integrator = DesignIntegratorV3(params, mandrel_diameters_mm=mandrel_D)
    result = integrator.optimize_full_wing(beta=0.9888, max_deflection_m=0.9)