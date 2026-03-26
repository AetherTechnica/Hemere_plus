import numpy as np
import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# =========================================================
# Pickleロード用定義（EOS モデル復元に必要）
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


class WingDesignParams:
    """
    翼設計に必要な入力パラメータ（W_total は入力しない）。

    W_total = W_pilot + W_fuselage + W_wing_secondary + W_spar(最適化結果)
    W_spar は設計変数（β）と積層最適化によって決まるため、最初に固定できない。
    """
    def __init__(self,
                 W_pilot_kg: float,
                 W_fuselage_kg: float,
                 W_wing_secondary_kg: float,
                 span_m: float,
                 V_flight_ms: float,
                 delta_allow_m: float,
                 sigma_allow_MPa: float,
                 mandrel_diameters_mm: np.ndarray,
                 rho_air: float = 1.154,
                 n_segments: int = 8):
        """
        Args:
            W_pilot_kg           : パイロット体重 [kg]
            W_fuselage_kg        : 機体（桁以外）固定重量 [kg]
            W_wing_secondary_kg  : 主翼二次構造（リブ・外皮等、桁以外）[kg]
            span_m               : 全スパン [m]
            V_flight_ms          : 飛行速度 [m/s]
            delta_allow_m        : 許容翼端たわみ [m]
            sigma_allow_MPa      : 許容縁応力 [MPa]
            mandrel_diameters_mm : マンドレル直径分布 [mm]（n_segments 個）
            rho_air              : 空気密度 [kg/m³]
            n_segments           : スパン分割数
        """
        self.W_pilot_kg          = W_pilot_kg
        self.W_fuselage_kg       = W_fuselage_kg
        self.W_wing_secondary_kg = W_wing_secondary_kg
        self.span_m              = span_m
        self.V_flight_ms         = V_flight_ms
        self.delta_allow_m       = delta_allow_m
        self.sigma_allow_MPa     = sigma_allow_MPa
        self.rho_air             = rho_air
        self.n_segments          = n_segments

        assert len(mandrel_diameters_mm) == n_segments, \
            f"mandrel_diameters_mm の長さ({len(mandrel_diameters_mm)})が n_segments({n_segments})と不一致"
        self.mandrel_diameters_mm = np.array(mandrel_diameters_mm, dtype=float)

    @property
    def W_fixed_kg(self):
        """桁重量を除く固定重量（パイロット + 機体 + 主翼二次構造）"""
        return self.W_pilot_kg + self.W_fuselage_kg + self.W_wing_secondary_kg


class DesignIntegratorV3:
    """
    空力-構造連成最適化による最適設計探索クラス。

    設計フロー:
        1. β をスイープ（Di 最小化）
        2. 各 β に対して:
               W_spar = 0 で初期化
               [収束ループ]
                   W_total = W_fixed + W_spar
                   空力計算: β, W_total → 揚力分布 L(y), Di
                   構造計算: L(y) - w_spar(y) → M(y)
                   EI 逆算: M(y), delta_allow → EI_req(y)
                   積層最適化: EI_req(y), D(y) → 積層構成, W_spar
               until W_spar 収束
        3. Di 最小の β* を返す
    """

    G = 9.80665

    def __init__(self, params: WingDesignParams):
        self.params    = params
        self.ei_est    = EIEstimator()
        self.layup_opt = LayupOptimizer()
        # 空力解析用の y グリッドは span と n_segments から決まる
        # （W_total によって変わらないので、ダミーで 1N を渡してグリッドだけ取得）
        _dummy_aero = self._build_aero(W_total_kg=1.0)
        self.y_m = _dummy_aero.y  # スパン方向座標 [m]
        self.struct = StructuralAnalyzer(self.y_m)
        self.dy    = self.y_m[1] - self.y_m[0] if len(self.y_m) > 1 else 1.0

    def _build_aero(self, W_total_kg: float) -> AerodynamicsAnalyzer:
        """W_total_kg を揚力目標に設定した空力解析器を生成する"""
        p = self.params
        aero_params = AircraftAeroParams(
            lift_target_N=W_total_kg * self.G,
            span_m=p.span_m,
            v_flight_ms=p.V_flight_ms,
            rho_air=p.rho_air,
            n_segments=p.n_segments,
        )
        return AerodynamicsAnalyzer(aero_params)

    def _solve_single_beta(self, beta: float,
                           max_iter: int = 10,
                           weight_tol: float = 0.005,
                           verbose_iter: bool = False) -> dict:
        """
        β を固定して「W_total → 空力 → EI_req → 積層 → W_spar」の収束ループを回す。

        収束変数は桁自重**分布** w_spar_dist [N/m]（スカラー近似ではない）。
        EI_req の計算に使う M と最終確認の M が一致するよう、
        積層結果の実際の重量分布を直接フィードバックする。

        Returns:
            dict or None
        """
        p = self.params

        # 初期桁自重分布: 固定重量の 5% を一様分布と仮定
        W_spar_init = p.W_fixed_kg * 0.05
        w_spar_dist = np.full(len(self.y_m), W_spar_init * self.G / (p.span_m / 2.0))
        wing_specs  = None

        for it in range(max_iter):
            # --- W_total（桁重量は半スパン積分 × 2）---
            W_spar_kg  = np.sum(w_spar_dist / self.G) * self.dy * 2.0
            W_total_kg = p.W_fixed_kg + W_spar_kg

            # --- 空力計算 ---
            aero = self._build_aero(W_total_kg)
            aero_res = aero.solve(beta)
            if not aero_res:
                return None

            # --- 正味荷重（揚力 - 実際の桁自重分布）---
            net_load = aero_res['lift_dist_N_m'] - w_spar_dist
            M_Nm     = self.struct.compute_bending_moment(net_load)

            # --- EI_req 逆算（たわみ制約から）---
            EI_req_Nmm2 = self.ei_est.calc_EI_req(M_Nm, self.y_m, p.delta_allow_m)
            EI_req_kgf  = EI_req_Nmm2 / self.G

            # --- 積層最適化 → 実際の桁自重分布を更新 ---
            wing_specs_new = []
            for ei_req, D in zip(EI_req_kgf, p.mandrel_diameters_mm):
                spec = self.layup_opt.find_min_weight_layup(ei_req, D)
                wing_specs_new.append(spec)

            actual_weight   = np.array([s['weight'] for s in wing_specs_new])  # [kg/m]
            w_spar_dist_new = actual_weight * self.G  # [N/m]

            # --- 収束チェック（重量分布の最大変化率）---
            rel_diff = np.max(np.abs(w_spar_dist_new - w_spar_dist)) / (np.max(w_spar_dist_new) + 1e-9)
            if verbose_iter:
                W_kg = np.sum(actual_weight) * self.dy * 2.0
                print(f"   iter {it+1}: W_spar={W_kg:.3f}kg, W_total={W_total_kg:.2f}kg, diff={rel_diff:.4f}")

            w_spar_dist = w_spar_dist_new
            wing_specs  = wing_specs_new
            if rel_diff < weight_tol:
                break  # 収束

        if wing_specs is None:
            return None

        # --- 最終構造解析（収束済み w_spar_dist と wing_specs を使う）---
        actual_weight  = np.array([s['weight']     for s in wing_specs])
        actual_EI_kgf  = np.array([s['EI_actual']  for s in wing_specs])
        actual_I_mm4   = np.array([s['I_mm4']       for s in wing_specs])
        actual_D_outer = np.array([s['D_outer_mm']  for s in wing_specs])

        W_spar_kg  = np.sum(actual_weight) * self.dy * 2.0
        W_total_kg = p.W_fixed_kg + W_spar_kg

        # 最終 M（収束後の自重分布で再計算）
        aero_final   = self._build_aero(W_total_kg)
        aero_res     = aero_final.solve(beta)
        net_load_fin = aero_res['lift_dist_N_m'] - actual_weight * self.G
        final_M      = self.struct.compute_bending_moment(net_load_fin)
        final_d, _   = self.struct.compute_deflection(final_M, actual_EI_kgf * self.G)
        final_sig    = self.struct.check_strength(final_M, actual_I_mm4, actual_D_outer)

        delta_tip  = float(final_d[-1])
        sigma_max  = float(np.max(final_sig))
        infeasible = sum(1 for s in wing_specs if not s['feasible'])

        feasible = (infeasible == 0 and
                    delta_tip  <= p.delta_allow_m   * 1.05 and
                    sigma_max  <= p.sigma_allow_MPa * 1.05)

        return {
            'beta'           : float(beta),
            'induced_drag_N' : float(aero_res['induced_drag_N']),
            'W_spar_kg'      : float(W_spar_kg),
            'W_total_kg'     : float(W_total_kg),
            'delta_tip_m'    : delta_tip,
            'sigma_max_MPa'  : sigma_max,
            'wing_specs'     : wing_specs,
            'feasible'       : feasible,
            'infeasible_segs': infeasible,
        }

    def optimize(self,
                 beta_min: float = 0.95,
                 beta_max: float = 1.01,
                 n_beta: int = 13,
                 verbose: bool = True) -> dict:
        """
        β をスイープして Di を最小化する最適設計を探索する。

        Args:
            beta_min, beta_max : β の探索範囲
            n_beta             : スイープ点数
            verbose            : 進捗表示

        Returns:
            dict:
                'beta_opt', 'Di_opt', 'W_spar_opt', 'W_total_opt',
                'delta_tip_m', 'sigma_max_MPa', 'wing_specs',
                'sweep_results'  ← 全 β の結果一覧
        """
        p = self.params
        betas = np.linspace(beta_min, beta_max, n_beta)

        if verbose:
            print("=" * 65)
            print(f" Hemere V3: β スイープ最適化")
            print(f" スパン={p.span_m}m  V={p.V_flight_ms}m/s")
            print(f" W_fixed={p.W_fixed_kg:.1f} kg  δ_allow={p.delta_allow_m*1000:.0f}mm")
            print("=" * 65)
            print(f" {'beta':>8} | {'Di [N]':>9} | {'W_spar[kg]':>11} | {'W_total[kg]':>12} | {'delta[mm]':>10} | {'σ[MPa]':>8} | OK?")
            print(" " + "-" * 75)

        sweep_results = []
        for beta in betas:
            res = self._solve_single_beta(beta, verbose_iter=False)
            if res is None:
                continue
            sweep_results.append(res)

            if verbose:
                ok = "YES" if res['feasible'] else " no"
                print(f" {beta:>8.4f} | {res['induced_drag_N']:>9.4f} | "
                      f"{res['W_spar_kg']:>10.3f}kg | "
                      f"{res['W_total_kg']:>11.3f}kg | "
                      f"{res['delta_tip_m']*1000:>8.1f}mm | "
                      f"{res['sigma_max_MPa']:>7.1f} | {ok}")

        # 実行可能解の中で Di が最小のものを選ぶ
        feasible_results = [r for r in sweep_results if r['feasible']]
        if not feasible_results:
            if verbose:
                print("\n [ERROR] 実行可能解が見つかりません。パラメータを確認してください。")
            # 実行不可能でも最小 Di を返す（デバッグ用）
            best = min(sweep_results, key=lambda r: r['induced_drag_N'])
        else:
            best = min(feasible_results, key=lambda r: r['induced_drag_N'])

        best['sweep_results'] = sweep_results

        if verbose:
            print(" " + "-" * 75)
            print(f"\n 最適解: beta*={best['beta']:.4f},  Di*={best['induced_drag_N']:.4f} N")
            print(f"         W_spar={best['W_spar_kg']:.3f} kg,  W_total={best['W_total_kg']:.3f} kg")
            print(f"         delta_tip={best['delta_tip_m']*1000:.1f} mm  (制約: {p.delta_allow_m*1000:.0f} mm)")
            print(f"         sigma_max={best['sigma_max_MPa']:.1f} MPa  (制約: {p.sigma_allow_MPa:.0f} MPa)")
            print("=" * 65)

        return best


if __name__ == "__main__":
    # opt-ATLAS 条件3（高速ショートスパン）での検証
    # 期待: beta* ≈ 0.9888, Di* ≈ 6.229N（W_total が同一の場合）
    params = WingDesignParams(
        W_pilot_kg          = 55.0,
        W_fuselage_kg       = 20.0,
        W_wing_secondary_kg = 8.0,
        span_m              = 22.0,
        V_flight_ms         = 11.0,
        delta_allow_m       = 0.9,
        sigma_allow_MPa     = 300.0,
        mandrel_diameters_mm= np.linspace(90, 40, 8),
        n_segments          = 8,
    )

    integrator = DesignIntegratorV3(params)
    result = integrator.optimize(beta_min=0.97, beta_max=1.01, n_beta=9)
