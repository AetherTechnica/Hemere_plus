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
from src.optimization.snap_optimizer import SnapOptimizer
from src.structural.structural_analyzer import StructuralAnalyzer
from src.core.ei_estimator import EIEstimator

class DesignIntegratorV3:
    def __init__(self, aero_params: AircraftAeroParams):
        self.aero = AerodynamicsAnalyzer(aero_params)
        self.snap_opt = SnapOptimizer()
        self.struct = StructuralAnalyzer(self.aero.y)
        self.ei_est = EIEstimator()
        self.g = 9.80665

    def optimize_full_wing(self, beta=0.9, n_mandrel_groups=3, max_deflection_m=1.5):
        # --- Step 1: 空力解析 ---
        aero_res = self.aero.solve(beta=beta)
        if not aero_res: return None

        # --- Step 2: 必要剛性の逆算 (EIEstimator) ---
        temp_moment = self.struct.compute_bending_moment(aero_res['lift_dist_N_m'])
        target_ei_dist = self.ei_est.calc_EI_req(temp_moment, self.aero.y, max_deflection_m)

        # --- Step 3: セクション最適化 (グループ化) ---
        print(f"\n{'='*60}\n Hemere V3: Physical Integration Loop\n{'='*60}")
        
        wing_specs = []
        n_sections = len(target_ei_dist)
        group_size = int(np.ceil(n_sections / n_mandrel_groups))
        
        for g in range(n_mandrel_groups):
            start, end = g * group_size, min((g + 1) * group_size, n_sections)
            if start >= n_sections: break
            
            # グループ代表直径をAIから取得 (N*mm² -> kgf*mm² 変換)
            group_max_ei_kgf = np.max(target_ei_dist[start:end]) / self.g
            ideal_r, _ = self.snap_opt.predict_ideal_spec(group_max_ei_kgf)

            print(f"\n[Group {g+1}] Recommended Diameter: {ideal_r:.1f}mm")

            for i in range(start, end):
                # snap_optimizer は kgf*mm² を期待するので変換
                solutions = self.snap_opt.solve(target_ei_dist[i] / self.g)
                if solutions:
                    wing_specs.append(solutions[0])  # 最軽量解（best）を採用

        # --- Step 4: 答え合わせ (最終構造解析) ---
        # snap_optimizer の辞書キー: 'EI', 'Weight', 'Diameter', 'I_mm4', 'D_outer_mm'
        actual_ei          = np.array([s['EI']         for s in wing_specs])
        actual_diameter    = np.array([s['Diameter']   for s in wing_specs])
        actual_weight_dist = np.array([s['Weight']     for s in wing_specs])
        actual_I_mm4       = np.array([s['I_mm4']      for s in wing_specs])
        actual_D_outer_mm  = np.array([s['D_outer_mm'] for s in wing_specs])

        # 正味荷重を再計算 (揚力 - 桁自重)
        net_load = aero_res['lift_dist_N_m'] - (actual_weight_dist * self.g)

        # 再積分 (EI: kgf*mm² -> N*mm² 変換して渡す)
        final_moment = self.struct.compute_bending_moment(net_load)
        final_deflection, _ = self.struct.compute_deflection(final_moment, actual_ei * self.g)
        # check_strength は I_mm4 と D_outer_mm を直接使う（E固定を廃止）
        final_stress = self.struct.check_strength(final_moment, actual_I_mm4, actual_D_outer_mm)

        # --- 結果表示 ---
        print(f"\n{'='*60}\n FINAL VERIFICATION\n{'='*60}")
        print(f" Target Deflection: {max_deflection_m:.3f} m")
        print(f" Actual Deflection: {final_deflection[-1]:.3f} m")
        print(f" Max Fiber Stress : {np.max(final_stress):.1f} MPa")
        print(f" Total Wing Weight: {np.sum(actual_weight_dist * (self.aero.y[1]-self.aero.y[0])) * 2:.3f} kg")
        print(f"{'='*60}")

        if final_deflection[-1] > max_deflection_m * 1.05:
            print(" [WARNING] Deflection exceeds target! Increase safety factor or target EI.")

if __name__ == "__main__":
    params = AircraftAeroParams(
        lift_target_N=100.0 * 9.81, # 85kg
        span_m=34.0,
        v_flight_ms=7.0,
        rho_air=1.154,
        n_segments=15
    )
    DesignIntegratorV3(params).optimize_full_wing(beta=0.95, max_deflection_m=2.5)