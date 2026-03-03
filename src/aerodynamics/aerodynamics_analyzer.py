import numpy as np
from dataclasses import dataclass

@dataclass
class AircraftAeroParams:
    """空力計算に必要な機体・環境パラメータ"""
    lift_target_N: float  # 必要揚力 (Total Lift)
    span_m: float        # 翼幅
    v_flight_ms: float   # 飛行速度
    rho_air: float       # 空気密度
    n_segments: int      # 分割数 (N)

class AerodynamicsAnalyzer:
    """
    TR-797手法に基づく空力計算クラス
    揚力分布、誘導抗力、循環分布を算出する
    """
    def __init__(self, params: AircraftAeroParams):
        self.p = params
        self.le = params.span_m / 2.0  # 半スパン
        
        # 離散化（等間隔分割）
        # opt_ATLASのロジックに従い、制御点と渦の配置を計算
        self.delta_S = np.ones(self.p.n_segments) * (self.le / self.p.n_segments / 2.0)
        self.y = np.linspace(self.delta_S[0], self.le - self.delta_S[-1], self.p.n_segments)
        self.z = np.zeros(self.p.n_segments)
        self.phi = np.zeros(self.p.n_segments) # 上反角・捻り角 (平面翼想定)

    def _compute_influence_matrix(self):
        """Q行列（誘導速度係数行列）を生成する"""
        N = self.p.n_segments
        y = self.y
        z = self.z
        phi = self.phi
        dS = self.delta_S

        # 行列計算用のブロードキャスト準備
        y_col = y[:, np.newaxis]
        z_col = z[:, np.newaxis]
        phi_col = phi[:, np.newaxis]

        # 相対座標の計算 (MATLABコードの ydach, zdash 等に対応)
        ydash = (y_col - y) * np.cos(phi) + (z_col - z) * np.sin(phi)
        zdash = -(y_col - y) * np.sin(phi) + (z_col - z) * np.cos(phi)
        y2dash = (y_col + y) * np.cos(phi) + (z_col - z) * np.sin(phi)
        z2dash = -(y_col + y) * np.sin(phi) + (z_col - z) * np.cos(phi)

        R_plus = (ydash - dS)**2 + zdash**2
        R_minus = (ydash + dS)**2 + zdash**2
        Rdash_plus = (y2dash + dS)**2 + z2dash**2
        Rdash_minus = (y2dash - dS)**2 + z2dash**2

        # Q行列の項計算
        term1 = (-(ydash - dS) / R_plus + (ydash + dS) / R_minus) * np.cos(phi_col - phi)
        term2 = (-zdash / R_plus + zdash / R_minus) * np.sin(phi_col - phi)
        term3 = (-(y2dash - dS) / Rdash_minus + (y2dash + dS) / Rdash_plus) * np.cos(phi_col + phi)
        term4 = (-z2dash / Rdash_minus + z2dash / Rdash_plus) * np.sin(phi_col + phi)

        Q = (1.0 / (2.0 * np.pi)) * (term1 + term2 + term3 + term4)
        return Q

    def solve(self, beta: float):
        """
        指定されたbetaに対して、誘導抗力と循環分布を解く
        Args:
            beta: モーメント低減率 (0.8 ~ 1.0程度)
        Returns:
            dict: {誘導抗力, 循環分布, 揚力分布, 効率}
        """
        N = self.p.n_segments
        Q = self._compute_influence_matrix()
        
        # 正規化変数
        delta_sigma = self.delta_S / self.le
        eta = self.y / self.le
        q_mat = Q * self.le
        
        # 行列Aの生成 (A_ij = pi * q_ij * delta_sigma_i)
        A = np.pi * q_mat * delta_sigma[:, np.newaxis]
        
        # 連立方程式の構築
        # [A+A', -c, -b; -c', 0, 0; -b', 0, 0] * [g; mu1; mu2] = [0; -1; -beta]
        c_vec = 2.0 * delta_sigma
        b_vec = (3.0 * np.pi / 2.0) * delta_sigma * eta
        
        sys_mat = np.zeros((N + 2, N + 2))
        sys_mat[:N, :N] = A + A.T
        sys_mat[:N, N] = -c_vec
        sys_mat[:N, N+1] = -b_vec
        sys_mat[N, :N] = -c_vec
        sys_mat[N+1, :N] = -b_vec
        
        rhs = np.zeros(N + 2)
        rhs[N] = -1.0
        rhs[N+1] = -beta
        
        # 求解
        try:
            sol = np.linalg.solve(sys_mat, rhs)
            g = sol[:N]
        except np.linalg.LinAlgError:
            return None

        # 物理量への復元
        # Gamma = (L / (2 * le * rho * U)) * g
        gamma_const = self.p.lift_target_N / (2.0 * self.le * self.p.rho_air * self.p.v_flight_ms)
        gamma = gamma_const * g
        
        # 揚力分布 [N/m] = rho * U * Gamma
        lift_dist = self.p.rho_air * self.p.v_flight_ms * gamma
        
        # 誘導抗力 Di
        # Di_ellipse = L^2 / (2 * pi * rho * U^2 * le^2)
        di_ellipse = self.p.lift_target_N**2 / (2.0 * np.pi * self.p.rho_air * (self.p.v_flight_ms**2) * (self.le**2))
        efficiency_inv = g.T @ A @ g
        di = efficiency_inv * di_ellipse
        
        return {
            "induced_drag_N": di,
            "gamma": gamma,
            "lift_dist_N_m": lift_dist,
            "span_y": self.y,
            "efficiency": 1.0 / efficiency_inv
        }

# --- テスト実行用 ---
if __name__ == "__main__":
    test_params = AircraftAeroParams(
        lift_target_N=100.0 * 9.81, # 100kg
        span_m=30.0,
        v_flight_ms=7.5,
        rho_air=1.154,
        n_segments=100
    )
    analyzer = AerodynamicsAnalyzer(test_params)
    results = analyzer.solve(beta=0.9)
    if results:
        print(f"Induced Drag: {results['induced_drag_N']:.4f} N")
        print(f"Efficiency: {results['efficiency']:.4f}")