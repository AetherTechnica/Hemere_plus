import numpy as np
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

class StructuralAnalyzer:
    """
    荷重分布と剛性分布から、梁の力学量（モーメント、たわみ、応力）を計算するクラス
    """
    def __init__(self, span_y: np.ndarray):
        self.y = span_y
        self.dy = span_y[1] - span_y[0] if len(span_y) > 1 else 0
        self.n = len(span_y)

    def compute_bending_moment(self, net_load_N_m: np.ndarray):
        """正味荷重分布から曲げモーメント分布を算出（翼端から積分）"""
        moment = np.zeros(self.n)
        for i in range(self.n):
            dist = self.y[i:] - self.y[i]
            moment[i] = np.trapz(net_load_N_m[i:] * dist, self.y[i:])
        return moment

    def compute_deflection(self, moment_Nm: np.ndarray, ei_dist_Nmm2: np.ndarray):
        """モーメントと剛性からたわみ分布を算出（翼根から積分）"""
        # 剛性単位換算: Nmm^2 -> Nm^2
        ei_Nm2 = ei_dist_Nmm2 * 1e-6
        curvature = moment_Nm / (ei_Nm2 + 1e-9)
        
        # 傾斜角 θ = ∫ κ dy (翼根で θ=0)
        theta = cumtrapz(curvature, self.y, initial=0)
        # たわみ δ = ∫ θ dy (翼根で δ=0)
        deflection = cumtrapz(theta, self.y, initial=0)
        
        return deflection, theta

    def check_strength(self, moment_Nm: np.ndarray, diameter_mm: np.ndarray, ei_dist_Nmm2: np.ndarray):
        """最大縁応力を計算 [MPa]"""
        e_modulus = 100e9 # 100 GPa
        i_dist_m4 = (ei_dist_Nmm2 * 1e-6) / e_modulus
        c_m = (diameter_mm / 1000.0) / 2.0
        stress = (moment_Nm * c_m) / (i_dist_m4 + 1e-12)
        return stress / 1e6

if __name__ == "__main__":
    # 簡易テスト
    y = np.linspace(0, 10, 50)
    load = np.full_like(y, 10)
    ei = np.full_like(y, 1e10)
    analyzer = StructuralAnalyzer(y)
    M = analyzer.compute_bending_moment(load)
    delta, _ = analyzer.compute_deflection(M, ei)
    print(f"Test Result: Tip Deflection = {delta[-1]:.3f} m")