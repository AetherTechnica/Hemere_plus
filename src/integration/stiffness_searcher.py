import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- Pickle対策 ---
def add_physics_features(X):
    log_ei = X[:, 0]; r = X[:, 1]
    return np.column_stack((X, log_ei - 3 * np.log10(r + 1e-9)))
class PhysicsFeatureEngineer:
    def transform(self, X): return add_physics_features(X)

from src.aerodynamics.aerodynamics_analyzer import AerodynamicsAnalyzer, AircraftAeroParams
from src.optimization.snap_optimizer import SnapOptimizer
from src.structural.structural_analyzer import StructuralAnalyzer

class StiffnessSearcherV2:
    def __init__(self, aero_params: AircraftAeroParams):
        self.aero = AerodynamicsAnalyzer(aero_params)
        self.snap_opt = SnapOptimizer()
        self.struct = StructuralAnalyzer(self.aero.y)
        self.max_deflection = 1.8
        self.gravity = 9.80665

        # 1. データの物理的限界を定義 (kgf*mm^2 単位)
        # 学習データ(D=130mm, 0度層多数)から計算される限界は約10.8 (6.3e10)
        self.LOG_EI_MAX_KGF = 10.8 
        self.LOG_EI_MIN_KGF = 8.0

        # 2. 重量推算の平滑化 (kind='linear' で外挿による負の値を防ぐ)
        print(f"Pre-calculating Global Pareto (Range: 10^{self.LOG_EI_MIN_KGF} - 10^{self.LOG_EI_MAX_KGF} kgf*mm^2)")
        self._log_ei_samples_kgf = np.linspace(self.LOG_EI_MIN_KGF, self.LOG_EI_MAX_KGF, 100)
        self._weight_samples = [self.snap_opt.predict_ideal_spec(10**lei)[1] for lei in self._log_ei_samples_kgf]
        
        # 'linear' にすることで、範囲外でも負の値に跳ねるのを防ぐ
        self.f_weight = interp1d(self._log_ei_samples_kgf, self._weight_samples, 
                                 kind='linear', fill_value="extrapolate")

    def optimize_ei_distribution(self, initial_beta=0.9):
        aero_res = self.aero.solve(beta=initial_beta)
        moment_dist_Nm = self.struct.compute_bending_moment(aero_res['lift_dist_N_m'])
        
        # 初期値 (Nmm^2)
        ei_init_Nmm2 = np.maximum(moment_dist_Nm * 1e6, 1e8) * 2.0
        x0 = np.log10(ei_init_Nmm2)
        
        # 探索境界を物理的限界 (Nmm^2) に設定
        # 例: 10^10.8 (kgf) * 9.8 => 10^11.79 (Nmm^2)
        upper_bound_Nmm2 = self.LOG_EI_MAX_KGF + np.log10(self.gravity)

        def objective(log_ei_arr_Nmm2):
            log_ei_arr_kgfmm2 = log_ei_arr_Nmm2 - np.log10(self.gravity)
            section_weights = self.f_weight(log_ei_arr_kgfmm2)
            return np.sum(np.maximum(section_weights, 0.05) * self.struct.dy) # 重量は最低50g/m

        def constraint_deflection(log_ei_arr_Nmm2):
            ei_dist_Nmm2 = 10**log_ei_arr_Nmm2
            delta, _ = self.struct.compute_deflection(moment_dist_Nm, ei_dist_Nmm2)
            return self.max_deflection - delta[-1]

        res = minimize(
            objective, x0, method='SLSQP',
            constraints={'type': 'ineq', 'fun': constraint_deflection},
            bounds=[(8.0, upper_bound_Nmm2)] * len(x0),
            options={'maxiter': 100}
        )

        # 収束しなくても物理的限界内でクリップした値を返す
        opt_ei = 10**np.clip(res.x, 8.0, upper_bound_Nmm2)
        print(f"Optimal EI Search Done. Est. Weight: {objective(np.log10(opt_ei)):.3f} kg")
        return opt_ei

if __name__ == "__main__":
    from src.aerodynamics.aerodynamics_analyzer import AircraftAeroParams
    params = AircraftAeroParams(
        lift_target_N=85.0 * 9.80665, span_m=24.0, 
        v_flight_ms=7.0, rho_air=1.154, n_segments=15
    )
    searcher = StiffnessSearcherV2(params)
    searcher.optimize_ei_distribution()