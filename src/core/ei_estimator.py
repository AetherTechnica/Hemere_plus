import numpy as np
from src.structural.structural_analyzer import StructuralAnalyzer


class EIEstimator:
    """
    許容たわみから必要曲げ剛性 EI_req(y) を逆算するクラス。

    アルゴリズム:
        1. 形状関数: EI_shape(y) = |M(y)| / max(|M|)  [無次元化]
        2. EI_req(y) = alpha * EI_shape(y)
        3. alpha を二分法で解: delta_tip(alpha) = delta_allow
    """

    def calc_EI_req(self,
                    M_Nm: np.ndarray,
                    y_m: np.ndarray,
                    delta_allow_m: float,
                    tol_rel: float = 1e-5,
                    max_iter: int = 100) -> np.ndarray:
        """
        許容たわみ delta_allow から必要 EI 分布を計算。

        Args:
            M_Nm          : 曲げモーメント分布 [N*m]
            y_m           : スパン方向座標配列 [m]（翼根=0, 翼端=L）
            delta_allow_m : 許容翼端たわみ [m]
            tol_rel       : 収束判定の相対許容誤差（デフォルト 1e-5）
            max_iter      : 二分法の最大反復回数（デフォルト 100）

        Returns:
            EI_req_Nmm2   : 必要曲げ剛性分布 [N*mm^2]
        """
        M_abs = np.abs(M_Nm)
        M_max = np.max(M_abs)

        # モーメントがほぼゼロ（無荷重）の場合
        if M_max < 1e-10:
            return np.ones(len(M_Nm)) * 1e6  # 最小剛性 [N*mm^2]

        # 無次元形状関数
        M_shape = M_abs / M_max

        def calc_delta_tip(alpha_Nm2: float) -> float:
            """
            スケール係数 alpha [N*m^2] に対する翼端たわみ [m] を計算。
            EI_Nm2(y) = alpha * M_shape(y) のたわみ積分。
            """
            EI_Nm2 = alpha_Nm2 * M_shape
            # 曲率 κ = M / EI [1/m]（EI=0 を防ぐため微小値を加算）
            curvature = M_abs / (EI_Nm2 + 1e-30)
            # 二重積分でたわみを求める（翼根で θ=0, δ=0）
            theta = StructuralAnalyzer._cumtrapz(curvature, y_m)
            delta = StructuralAnalyzer._cumtrapz(theta, y_m)
            return delta[-1]

        # ブラケット設定
        # alpha 小 → EI 小 → delta 大（柔らかい）
        # alpha 大 → EI 大 → delta 小（硬い）
        alpha_lo = 1e0    # [N*m^2] 非常に柔らかい
        alpha_hi = 1e12   # [N*m^2] 非常に硬い

        # ブラケットが成立するか確認
        delta_lo = calc_delta_tip(alpha_lo)
        delta_hi = calc_delta_tip(alpha_hi)

        if delta_lo < delta_allow_m:
            # 最も柔らかい設定でも delta が許容値以下 → alpha_lo をそのまま返す
            EI_req_Nm2 = alpha_lo * M_shape
            return EI_req_Nm2 * 1e6

        if delta_hi > delta_allow_m:
            # 最も硬い設定でも delta が許容値超 → alpha_hi をそのまま返す
            EI_req_Nm2 = alpha_hi * M_shape
            return EI_req_Nm2 * 1e6

        # 二分法
        alpha_mid = alpha_lo
        for _ in range(max_iter):
            alpha_mid = (alpha_lo + alpha_hi) / 2.0
            delta_mid = calc_delta_tip(alpha_mid)

            if abs(delta_mid - delta_allow_m) / delta_allow_m < tol_rel:
                break  # 収束

            if delta_mid > delta_allow_m:
                # delta が大きすぎる → EI を増やす（alpha を大きく）
                alpha_lo = alpha_mid
            else:
                # delta が小さすぎる → EI を減らす（alpha を小さく）
                alpha_hi = alpha_mid

        alpha_opt = alpha_mid
        EI_req_Nm2  = alpha_opt * M_shape
        EI_req_Nmm2 = EI_req_Nm2 * 1e6  # N*m^2 -> N*mm^2
        return EI_req_Nmm2


if __name__ == "__main__":
    # 簡易テスト: 等分布荷重カンチレバーの解析解と比較
    # δ_tip = w*L^4 / (8*EI)
    # → EI = w*L^4 / (8*δ_tip)
    L   = 16.0     # [m]
    w   = 100.0    # [N/m]
    delta_allow = 0.8192  # [m]

    # 解析解の EI
    EI_analytic_Nm2 = w * L**4 / (8.0 * delta_allow)
    print(f"--- EIEstimator テスト ---")
    print(f"等分布荷重: {w} N/m, スパン: {L} m, 許容たわみ: {delta_allow} m")
    print(f"解析解 EI : {EI_analytic_Nm2:.4e} N*m^2 = {EI_analytic_Nm2*1e6:.4e} N*mm^2")

    y = np.linspace(0, L, 200)
    M = 0.5 * w * (L - y) ** 2  # 等分布荷重のモーメント分布 [N*m]

    est = EIEstimator()
    EI_req = est.calc_EI_req(M, y, delta_allow)

    # 逆検証: 求めた EI で δ を計算
    from src.structural.structural_analyzer import StructuralAnalyzer
    sa = StructuralAnalyzer(y)
    d, _ = sa.compute_deflection(M, EI_req)

    print(f"\n--- 逆検証 ---")
    print(f"計算 delta_tip : {d[-1]*1000:.3f} mm")
    print(f"期待 delta_tip : {delta_allow*1000:.3f} mm")
    print(f"誤差           : {abs(d[-1] - delta_allow) / delta_allow * 100:.5f} %")
