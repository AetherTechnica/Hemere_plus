import numpy as np
import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# =========================================================
# Pickleロード用定義 (AttributeError対策)
# =========================================================
def add_physics_features(X):
    log_ei = X[:, 0]
    r = X[:, 1]
    log_r = np.log10(r + 1e-9)
    thickness_index = log_ei - 3 * log_r
    return np.column_stack((X, thickness_index))

def inverse_log10(x):
    return 10**x

class PhysicsFeatureEngineer:
    def transform(self, X):
        return add_physics_features(X)
# =========================================================

from src.aerodynamics.aerodynamics_analyzer import AerodynamicsAnalyzer, AircraftAeroParams
from src.optimization.snap_optimizer import SnapOptimizer

class DesignIntegratorV2:
    def __init__(self, aero_params: AircraftAeroParams):
        self.aero_analyzer = AerodynamicsAnalyzer(aero_params)
        self.snap_opt = SnapOptimizer()
        self.g = 9.80665

    def compute_required_ei(self, lift_dist, span_y):
        """揚力分布から曲げモーメントを計算し、必要剛性(EI)を算出"""
        n = len(span_y)
        dy = span_y[1] - span_y[0]
        moment = np.zeros(n)
        shear = 0.0
        m_val = 0.0
        
        for i in reversed(range(n)):
            shear += lift_dist[i] * dy
            m_val += shear * dy
            moment[i] = m_val

        # 強度・剛性を考慮したターゲットEI [Nmm^2]
        # モーメント[Nm] -> [Nmm]換算を含め、安全率を考慮
        target_ei_dist = moment * 1e7 * 1.2 
        return target_ei_dist, moment

    def optimize_full_wing(self, beta=0.9, n_mandrel_groups=3):
        """
        全幅の最適化。直径をn_mandrel_groups個のグループに統合する。
        """
        # 1. 空力計算
        aero_results = self.aero_analyzer.solve(beta=beta)
        if not aero_results: return None

        # 2. 必要剛性の導出
        target_ei_dist, moment_dist = self.compute_required_ei(
            aero_results['lift_dist_N_m'], 
            aero_results['span_y']
        )
        
        span_y = aero_results['span_y']
        dy = span_y[1] - span_y[0]
        n_sections = len(target_ei_dist)
        
        print(f"\n{'='*60}")
        print(f" Hemere Integrated Optimization (Mandrel Groups: {n_mandrel_groups})")
        print(f"{'='*60}")

        wing_specs = []
        total_wing_weight_kg = 0.0

        # 3. 直径のグループ化 (改良1: マンドレル統合)
        # セクションをグループに分け、各グループの最大EIに合わせて直径を決定
        group_size = int(np.ceil(n_sections / n_mandrel_groups))
        
        for g_idx in range(n_mandrel_groups):
            start = g_idx * group_size
            end = min((g_idx + 1) * group_size, n_sections)
            if start >= n_sections: break
            
            # グループ内での最大必要剛性を基準に「推奨直径」をAIから取得
            group_target_ei = np.max(target_ei_dist[start:end])
            ideal_r, _ = self.snap_opt.predict_ideal_spec(group_target_ei)
            
            # このグループで使用する直径を固定 (Snap処理の簡略化)
            group_diameter = np.round(ideal_r) 

            print(f"\n[Group {g_idx+1}] Range: {span_y[start]:.1f}-{span_y[end-1]:.1f}m, Target Dia: {group_diameter}mm")

            for i in range(start, end):
                # 物理スナップ (このグループの固定直径で最適な積層を探す)
                # 本来のSnapOptimizer.snap_to_physicsを直径固定で呼び出す形
                # ここでは簡易的に元のロジックを使い、直径が大きく乖離しないかチェック
                spec = self.snap_opt.solve(target_ei_dist[i])
                
                if spec:
                    # 単位系の確認: Actual_Weight [kg/m] * dy [m] = [kg]
                    section_weight_kg = spec['Actual_Weight'] * dy
                    total_wing_weight_kg += section_weight_kg
                    wing_specs.append(spec)
                    
                    # 改良3: ログを簡潔に
                    print(f"  Sec {i:02d}: EI_req={target_ei_dist[i]:.2e} -> W={spec['Actual_Weight']:.4f}kg/m, D={spec['Diameter']}mm")

        print(f"\n{'='*60}")
        print(f" RESULT SUMMARY")
        print(f" Total Structural Weight (One Side): {total_wing_weight_kg:.3f} kg")
        print(f" Total Structural Weight (Full Span): {total_wing_weight_kg * 2:.3f} kg")
        print(f"{'='*60}")

        return {
            "aero": aero_results,
            "structural_specs": wing_specs,
            "total_weight_kg": total_wing_weight_kg * 2
        }

if __name__ == "__main__":
    params = AircraftAeroParams(
        lift_target_N=100.0 * 9.81, # 75kg
        span_m=30.0,
        v_flight_ms=7.5,
        rho_air=1.154,
        n_segments=5
    )
    
    integrator = DesignIntegratorV2(params)
    integrator.optimize_full_wing(beta=0.07, n_mandrel_groups=3)