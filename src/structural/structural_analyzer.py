import numpy as np
# scipy 不要（pip 遮断環境対応）: cumtrapz を純 NumPy で実装済み

class StructuralAnalyzer:
    """
    荷重分布と剛性分布から、梁の力学量（モーメント、たわみ、応力）を計算するクラス
    """

    def __init__(self, span_y: np.ndarray):
        self.y = span_y
        self.dy = span_y[1] - span_y[0] if len(span_y) > 1 else 0
        self.n = len(span_y)

    @staticmethod
    def _cumtrapz(y: np.ndarray, x: np.ndarray, initial: float = 0.0) -> np.ndarray:
        """
        scipy.integrate.cumulative_trapezoid の純 NumPy 代替実装。
        各点における累積台形積分を返す（initial 値から始まる）。
        """
        dx = np.diff(x)
        trapz = (y[:-1] + y[1:]) * dx / 2.0
        return np.concatenate([[initial], np.cumsum(trapz)])

    def compute_bending_moment(self, net_load_N_m: np.ndarray) -> np.ndarray:
        """
        正味荷重分布から曲げモーメント分布を算出（翼端から積分）。

        Args:
            net_load_N_m: 正味荷重分布 [N/m]（揚力 - 自重）
        Returns:
            moment [N*m]
        """
        moment = np.zeros(self.n)
        for i in range(self.n - 1):
            dist = self.y[i:] - self.y[i]
            moment[i] = np.trapz(net_load_N_m[i:] * dist, self.y[i:])
        moment[-1] = 0.0
        return moment

    def compute_deflection(self, moment_Nm: np.ndarray, ei_dist_Nmm2: np.ndarray):
        """
        モーメントと剛性からたわみ分布を算出（翼根から積分）。

        Args:
            moment_Nm    : 曲げモーメント分布 [N*m]
            ei_dist_Nmm2 : 曲げ剛性分布 [N*mm^2]
        Returns:
            (deflection [m], theta [rad])
        """
        # 剛性単位換算: N*mm^2 -> N*m^2
        ei_Nm2 = ei_dist_Nmm2 * 1e-6
        curvature = np.abs(moment_Nm) / (ei_Nm2 + 1e-9)  # [1/m]

        # 傾斜角 θ = ∫ κ dy（翼根で θ=0）
        theta = self._cumtrapz(curvature, self.y)
        # たわみ δ = ∫ θ dy（翼根で δ=0）
        deflection = self._cumtrapz(theta, self.y)

        return deflection, theta

    def check_strength(self, moment_Nm: np.ndarray,
                       I_dist_mm4: np.ndarray,
                       D_outer_dist_mm: np.ndarray) -> np.ndarray:
        """
        最大縁応力を計算 [MPa]。

        σ = M * c / I  （c = 外半径）

        Args:
            moment_Nm      : 曲げモーメント分布 [N*m]
            I_dist_mm4     : 断面二次モーメント分布 [mm^4]（spar_calculator の返値）
            D_outer_dist_mm: 外径分布 [mm]（spar_calculator の返値）
        Returns:
            stress [MPa]
        """
        I_m4 = I_dist_mm4 * 1e-12          # mm^4 -> m^4
        c_m  = D_outer_dist_mm / 1000.0 / 2.0  # 外半径 [m]
        stress = np.abs(moment_Nm) * c_m / (I_m4 + 1e-18)
        return stress / 1e6  # MPa


if __name__ == "__main__":
    # 簡易テスト: 等分布荷重カンチレバー解析解との比較
    # delta_tip = w*L^4 / (8*EI)
    L = 10.0   # [m]
    w = 10.0   # [N/m]
    EI_Nm2 = 1e6   # [N*m^2]
    EI_Nmm2 = EI_Nm2 * 1e6

    y = np.linspace(0, L, 200)
    load = np.full_like(y, w)
    ei   = np.full_like(y, EI_Nmm2)

    analyzer = StructuralAnalyzer(y)
    M = analyzer.compute_bending_moment(load)
    delta, _ = analyzer.compute_deflection(M, ei)

    delta_analytic = w * L**4 / (8 * EI_Nm2)
    print(f"delta_tip 計算値: {delta[-1]:.6f} m")
    print(f"delta_tip 解析解: {delta_analytic:.6f} m")
    print(f"誤差: {abs(delta[-1]-delta_analytic)/delta_analytic*100:.4f}%")
