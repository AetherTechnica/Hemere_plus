import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- Pickle対策 ---
def add_physics_features(X):
    log_ei = X[:, 0]; r = X[:, 1]
    return np.column_stack((X, log_ei - 3 * np.log10(r + 1e-9)))
class PhysicsFeatureEngineer:
    def transform(self, X): return add_physics_features(X)
# ------------------

from src.integration.stiffness_searcher import StiffnessSearcherV2
from src.optimization.snap_optimizer import SnapOptimizer
from src.aerodynamics.aerodynamics_analyzer import AircraftAeroParams

class FinalDesignGenerator:
    def __init__(self, aero_params: AircraftAeroParams, max_deflection=1.8, fixed_wing_weight=10.0):
        self.searcher = StiffnessSearcherV2(aero_params)
        self.searcher.max_deflection = max_deflection
        self.snap_opt = SnapOptimizer()
        self.gravity = 9.80665
        self.fixed_wing_weight = fixed_wing_weight

    def generate_fixed_diameter(self, beta=0.9, mandrel_diameters=[100, 80, 60, 40, 30], safety_factor=1.2):
        """
        mandrel_diameters: 各セクションで使用するマンドレルの外径リスト [mm]
        """
        n_struct_sections = len(mandrel_diameters)
        
        # 1. 理想剛性分布の探索
        print(f"Searching for optimal stiffness distribution (N={self.searcher.aero.p.n_segments})...")
        opt_ei_full_Nmm2 = self.searcher.optimize_ei_distribution(initial_beta=beta)
        if opt_ei_full_Nmm2 is None: return

        # 2. セクション化
        n_full = len(opt_ei_full_Nmm2)
        group_size = int(np.ceil(n_full / n_struct_sections))
        
        print(f"\n{'='*95}")
        print(f" FINAL PRODUCTION SPECIFICATION (Fixed Diameters, SF: {safety_factor})")
        print(f"{'='*95}")
        print(f"{'Sec':<4} | {'Range[m]':<12} | {'EI_req[Nmm2]':<12} | {'D_fixed':<7} | {'W[kg/m]':<8} | {'Ply Config'}")
        print("-" * 95)

        dy_full = self.searcher.struct.dy
        struct_specs_full = []
        total_spar_weight = 0

        for m, D_fixed in enumerate(mandrel_diameters):
            idx_start = m * group_size
            idx_end = min((m + 1) * group_size, n_full)
            if idx_start >= n_full: break
            
            # 安全率を考慮した目標剛性
            section_target_ei_Nmm2 = np.mean(opt_ei_full_Nmm2[idx_start:idx_end]) * safety_factor
            target_ei_kgf = section_target_ei_Nmm2 / self.gravity
            
            # 直径を AI に選ばせず、D_fixed の前後 0mm でスナップ（実質固定）
            # snap_to_physics の探索範囲を非常に狭めることで固定を実現
            self.snap_opt.snap_range = 0.1 
            spec = self.snap_opt.snap_to_physics(target_ei_kgf, D_fixed)
            
            if not spec:
                # 指定直径で剛性が足りない場合、警告を出して失敗を表示
                print(f"{m:02d}   | {'ERROR':<12} | 指定された直径 {D_fixed}mm では剛性が不足します")
                spec = {'Actual_EI': 1e7/self.gravity, 'Actual_Weight': 1.0, 'Diameter': D_fixed, 'Ply_Config': 'INSUFFICIENT'}
            else:
                range_str = f"{self.searcher.aero.y[idx_start]:.1f}-{self.searcher.aero.y[idx_end-1]:.1f}"
                print(f"{m:02d}   | {range_str:<12} | {section_target_ei_Nmm2:.2e} | {spec['Diameter']:>7.1f} | {spec['Actual_Weight']:>8.4f} | {spec['Ply_Config']}")

            for _ in range(idx_start, idx_end):
                struct_specs_full.append(spec)
                total_spar_weight += spec['Actual_Weight'] * dy_full

        # 3. 最終検証
        actual_ei_Nmm2 = np.array([s['Actual_EI'] * self.gravity for s in struct_specs_full])
        actual_spar_weights = np.array([s['Actual_Weight'] for s in struct_specs_full])
        fixed_load_dist = (self.fixed_wing_weight / 2.0 / (self.searcher.aero.p.span_m / 2.0)) * self.gravity
        
        aero_res = self.searcher.aero.solve(beta=beta)
        net_load = aero_res['lift_dist_N_m'] - (actual_spar_weights * self.gravity) - fixed_load_dist
        
        final_moment = self.searcher.struct.compute_bending_moment(net_load)
        final_delta, _ = self.searcher.struct.compute_deflection(final_moment, actual_ei_Nmm2)
        
        print("-" * 95)
        print(f" Total Spar Weight (Full Wing): {total_spar_weight * 2:.3f} kg")
        print(f" Final Tip Deflection         : {final_delta[-1]:.3f} m (Target: {self.searcher.max_deflection}m)")
        print(f"{'='*95}")

if __name__ == "__main__":
    params = AircraftAeroParams(
        lift_target_N=106.0 * 9.80665, span_m=34.0, 
        v_flight_ms=7.2, rho_air=1.154, n_segments=100
    )
    
    # 5つのマンドレル径を自分で設定
    my_mandrels = [120.0, 100.0, 80.0, 65.0, 45.0]
    
    generator = FinalDesignGenerator(params, max_deflection=2.2, fixed_wing_weight=12.0)
    generator.generate_fixed_diameter(beta=0.9, mandrel_diameters=my_mandrels, safety_factor=1)