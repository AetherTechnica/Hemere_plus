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

    分割の考え方:
        n_aero : 空力計算・構造計算（曲げモーメント・たわみ）の離散化点数
                 大きいほど精度が上がる。デフォルト 100。
        n_spar : 桁径・積層最適化の分割数。
                 mandrel_diameters_mm の要素数と一致させる。デフォルト 8。
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
                 n_spar: int = 8,
                 n_aero: int = 100):
        """
        Args:
            W_pilot_kg           : パイロット体重 [kg]
            W_fuselage_kg        : 機体（桁以外）固定重量 [kg]
            W_wing_secondary_kg  : 主翼二次構造（リブ・外皮等、桁以外）[kg]
            span_m               : 全スパン [m]
            V_flight_ms          : 飛行速度 [m/s]
            delta_allow_m        : 許容翼端たわみ [m]
            sigma_allow_MPa      : 許容縁応力 [MPa]
            mandrel_diameters_mm : マンドレル直径分布 [mm]（n_spar 個）
            rho_air              : 空気密度 [kg/m³]
            n_spar               : 桁径・積層最適化の分割数（mandrel_diameters_mm の長さと一致）
            n_aero               : 空力・構造計算の分割数（精度向上のため n_spar より大きく取る）
        """
        self.W_pilot_kg          = W_pilot_kg
        self.W_fuselage_kg       = W_fuselage_kg
        self.W_wing_secondary_kg = W_wing_secondary_kg
        self.span_m              = span_m
        self.V_flight_ms         = V_flight_ms
        self.delta_allow_m       = delta_allow_m
        self.sigma_allow_MPa     = sigma_allow_MPa
        self.rho_air             = rho_air
        self.n_spar              = n_spar
        self.n_aero              = n_aero

        assert len(mandrel_diameters_mm) == n_spar, \
            f"mandrel_diameters_mm の長さ({len(mandrel_diameters_mm)})が n_spar({n_spar})と不一致"
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
                   空力計算（n_aero 点）: β, W_total → 揚力分布 L(y), Di
                   構造計算（n_aero 点）: L(y) - w_spar(y) → M(y)
                   EI 逆算（n_aero 点）: M(y), delta_allow → EI_req(y)
                   100→8 変換: 各桁セグメントの最大 EI_req を代表値として抽出
                   積層最適化（n_spar 点）: EI_req(y), D(y) → 積層構成, W_spar
                   8→100 展開: 積層結果をステップ関数で 100 点グリッドに展開
               until W_spar 収束
        3. Di 最小の β* を返す
    """

    G = 9.80665

    def __init__(self, params: WingDesignParams):
        self.params    = params
        self.ei_est    = EIEstimator()
        self.layup_opt = LayupOptimizer()

        # --- 空力・構造グリッド（n_aero 点）---
        _dummy_aero = self._build_aero(W_total_kg=1.0)
        self.y_m  = _dummy_aero.y               # 半スパン方向座標 [m], shape (n_aero,)
        self.dy   = self.y_m[1] - self.y_m[0]   # グリッド間隔 [m]（等間隔）
        self.struct = StructuralAnalyzer(self.y_m)

        # --- 桁セグメントグリッド（n_spar 点）---
        le = params.span_m / 2.0
        delta_s_spar = le / params.n_spar / 2.0
        self.y_spar  = np.linspace(delta_s_spar, le - delta_s_spar, params.n_spar)

        # --- 各 y_m 点が属する桁セグメントのインデックス（0 ~ n_spar-1）---
        # 隣接セグメントの中点を境界として分類
        spar_bounds = np.concatenate([
            [0.0],
            (self.y_spar[:-1] + self.y_spar[1:]) / 2.0,
            [le + 1e-9],
        ])
        self.spar_idx = np.searchsorted(spar_bounds[1:], self.y_m).clip(0, params.n_spar - 1)

    def _build_aero(self, W_total_kg: float) -> AerodynamicsAnalyzer:
        """W_total_kg を揚力目標に設定した空力解析器を生成する（n_aero 点）"""
        p = self.params
        aero_params = AircraftAeroParams(
            lift_target_N=W_total_kg * self.G,
            span_m=p.span_m,
            v_flight_ms=p.V_flight_ms,
            rho_air=p.rho_air,
            n_segments=p.n_aero,
        )
        return AerodynamicsAnalyzer(aero_params)

    def _solve_single_beta(self, beta: float,
                           max_iter: int = 10,
                           weight_tol: float = 0.005,
                           verbose_iter: bool = False) -> dict:
        """
        β を固定して「W_total → 空力 → EI_req → 積層 → W_spar」の収束ループを回す。

        空力・構造計算は n_aero 点で行い、積層最適化は n_spar 点（桁セグメント単位）で行う。
        積層結果をステップ関数で n_aero 点グリッドに展開してフィードバックする。

        Returns:
            dict or None
        """
        p = self.params

        # 翼二次構造の分布荷重 [N/m]（n_aero 点、半スパン均一分布）
        w_secondary_dist = np.full(
            len(self.y_m),
            p.W_wing_secondary_kg * self.G / (p.span_m / 2.0)
        )

        # 初期桁自重分布: 固定重量の 5% を均一と仮定
        W_spar_init = p.W_fixed_kg * 0.05
        w_spar_dist = np.full(len(self.y_m), W_spar_init * self.G / (p.span_m / 2.0))
        wing_specs  = None

        for it in range(max_iter):
            # --- W_total（半スパン積分 × 2）---
            W_spar_kg  = np.sum(w_spar_dist / self.G) * self.dy * 2.0
            W_total_kg = p.W_fixed_kg + W_spar_kg

            # --- 空力計算（n_aero 点）---
            aero = self._build_aero(W_total_kg)
            aero_res = aero.solve(beta)
            if not aero_res:
                return None

            # --- 正味荷重・曲げモーメント（n_aero 点）---
            net_load = aero_res['lift_dist_N_m'] - w_spar_dist - w_secondary_dist
            M_Nm     = self.struct.compute_bending_moment(net_load)

            # --- EI_req 逆算（n_aero 点）---
            EI_req_Nmm2 = self.ei_est.calc_EI_req(M_Nm, self.y_m, p.delta_allow_m)
            EI_req_kgf  = EI_req_Nmm2 / self.G

            # --- n_aero → n_spar：各桁セグメント内の最大 EI_req を代表値として使用 ---
            # 最大値を取ることで、そのセグメントで最も要求が厳しい点を満たす設計になる
            EI_req_spar = np.array([
                np.max(EI_req_kgf[self.spar_idx == i])
                if np.any(self.spar_idx == i) else self.layup_opt.MIN_EI_REQ
                for i in range(p.n_spar)
            ])

            # --- 積層最適化（n_spar 点）---
            wing_specs_new = [
                self.layup_opt.find_min_weight_layup(ei, D)
                for ei, D in zip(EI_req_spar, p.mandrel_diameters_mm)
            ]

            # --- n_spar → n_aero：ステップ関数で 100 点グリッドに展開 ---
            weight_spar       = np.array([s['weight'] for s in wing_specs_new])  # [kg/m]
            actual_weight_100 = weight_spar[self.spar_idx]                        # [n_aero]
            w_spar_dist_new   = actual_weight_100 * self.G                        # [N/m]

            # --- 収束チェック（重量分布の最大変化率）---
            rel_diff = (np.max(np.abs(w_spar_dist_new - w_spar_dist))
                        / (np.max(w_spar_dist_new) + 1e-9))
            if verbose_iter:
                W_kg = np.sum(actual_weight_100) * self.dy * 2.0
                print(f"   iter {it+1}: W_spar={W_kg:.3f}kg, "
                      f"W_total={W_total_kg:.2f}kg, diff={rel_diff:.4f}")

            w_spar_dist = w_spar_dist_new
            wing_specs  = wing_specs_new
            if rel_diff < weight_tol:
                break

        if wing_specs is None:
            return None

        # --- 最終構造解析（収束済み分布で再計算）---
        weight_spar  = np.array([s['weight']     for s in wing_specs])
        EI_kgf_spar  = np.array([s['EI_actual']  for s in wing_specs])
        I_mm4_spar   = np.array([s['I_mm4']      for s in wing_specs])
        D_outer_spar = np.array([s['D_outer_mm'] for s in wing_specs])

        # n_spar → n_aero 展開
        actual_weight_100 = weight_spar[self.spar_idx]
        EI_kgf_100        = EI_kgf_spar[self.spar_idx]
        I_mm4_100         = I_mm4_spar[self.spar_idx]
        D_outer_100       = D_outer_spar[self.spar_idx]

        W_spar_kg  = np.sum(actual_weight_100) * self.dy * 2.0
        W_total_kg = p.W_fixed_kg + W_spar_kg

        aero_final   = self._build_aero(W_total_kg)
        aero_res     = aero_final.solve(beta)
        net_load_fin = (aero_res['lift_dist_N_m']
                        - actual_weight_100 * self.G
                        - w_secondary_dist)
        final_M   = self.struct.compute_bending_moment(net_load_fin)
        final_d, _ = self.struct.compute_deflection(final_M, EI_kgf_100 * self.G)
        final_sig  = self.struct.check_strength(final_M, I_mm4_100, D_outer_100)

        delta_tip  = float(final_d[-1])
        sigma_max  = float(np.max(final_sig))
        infeasible = sum(1 for s in wing_specs if not s['feasible'])

        feasible = (infeasible == 0
                    and delta_tip <= p.delta_allow_m   * 1.05
                    and sigma_max <= p.sigma_allow_MPa * 1.05)

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
            print(f" n_aero={p.n_aero}  n_spar={p.n_spar}")
            print("=" * 65)
            print(f" {'beta':>8} | {'Di [N]':>9} | {'W_spar[kg]':>11} | "
                  f"{'W_total[kg]':>12} | {'delta[mm]':>10} | {'σ[MPa]':>8} | OK?")
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


# =============================================================================
# 可視化・結果レポート関数
# =============================================================================

# レイヤー情報: (コード, 材料・角度, 役割)
# ply_angles = [90, 90, 90, 50, 45, 40, 35, 30, 25, 20, 90]
# 90° = 全周巻き、50°〜20° = 集中積層（Cap）
LAYER_INFO = [
    ("L00", "24t  90° 全周", "内面固定(1ply)"),
    ("L01", "40t  90° 全周", "固定(2ply)   "),
    ("L02", "40t  90° 全周", "Base         "),
    ("L03", "40t  50° Cap ", "集中①       "),
    ("L04", "40t  45° Cap ", "集中②       "),
    ("L05", "40t  40° Cap ", "集中③       "),
    ("L06", "40t  35° Cap ", "集中④       "),
    ("L07", "40t  30° Cap ", "集中⑤       "),
    ("L08", "40t  25° Cap ", "集中⑥       "),
    ("L09", "40t  20° Cap ", "集中⑦       "),
    ("L10", "24t  90° 全周", "外面固定(1ply)"),
]


def print_layup_summary(result: dict) -> None:
    """
    最適解の積層構成をセグメントごとに表示する。

    Args:
        result: optimize() の戻り値
    """
    wing_specs = result['wing_specs']
    n_spar = len(wing_specs)
    sw = 8  # 各セグメント列幅

    print()
    print("=" * (34 + sw * n_spar))
    print(f"  最適解 (beta*={result['beta']:.4f},  Di*={result['induced_drag_N']:.4f} N) の積層構成")
    print(f"  W_spar={result['W_spar_kg']:.3f} kg,  W_total={result['W_total_kg']:.3f} kg")
    print("=" * (34 + sw * n_spar))

    # セグメントヘッダー
    header = f"  {'レイヤー':<16}{'役割':<15}"
    for i in range(n_spar):
        header += f"{'Seg'+str(i+1):>{sw}}"
    print(header)
    print("  " + "-" * (31 + sw * n_spar))

    # 各レイヤーの ply 数
    for li, (code, material, role) in enumerate(LAYER_INFO):
        row = f"  {code+' '+material:<31}"
        for spec in wing_specs:
            pc = spec['ply_counts'][li]
            mark = "*" if pc > 0 else " "
            row += f"{str(pc)+mark:>{sw}}"
        print(row)

    print("  " + "-" * (31 + sw * n_spar))

    # スペック行
    spec_rows = [
        ("EI [x1e9 kgf*mm2]", lambda s: f"{s['EI_actual']/1e9:>{sw}.3f}"),
        ("Weight [kg/m]",      lambda s: f"{s['weight']:>{sw}.4f}"),
        ("D_outer [mm]",       lambda s: f"{s['D_outer_mm']:>{sw}.1f}"),
        ("D/t",                lambda s: f"{s['D_t']:>{sw}.1f}"),
        ("OK?",                lambda s: f"{'YES':>{sw}}" if s['feasible'] else f"{'NO':>{sw}}"),
    ]
    for label, fmt in spec_rows:
        row = f"  {label:<31}"
        for spec in wing_specs:
            row += fmt(spec)
        print(row)

    print("=" * (34 + sw * n_spar))


def plot_beta_sweep(result: dict, save_path: str = 'beta_sweep_Di.png') -> None:
    """
    β スイープ結果の可視化（Di推移 + W_spar推移の2段グラフ）。

    上段: β vs 誘導抗力 Di（実行可能/不可を色分け、最適点を★でマーク）
    下段: β vs 桁重量 W_spar

    Args:
        result   : optimize() の戻り値
        save_path: 画像保存パス
    """
    try:
        import matplotlib
        matplotlib.rcParams['font.family'] = 'MS Gothic'
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib が見つかりません。グラフ出力をスキップします。")
        return

    sweep    = result['sweep_results']
    betas    = [r['beta']           for r in sweep]
    dis      = [r['induced_drag_N'] for r in sweep]
    wspar    = [r['W_spar_kg']      for r in sweep]
    feasible = [r['feasible']       for r in sweep]

    # 実行可能・不可能に分割
    b_ok = [b for b, f in zip(betas, feasible) if     f]
    d_ok = [d for d, f in zip(dis,   feasible) if     f]
    w_ok = [w for w, f in zip(wspar, feasible) if     f]
    b_ng = [b for b, f in zip(betas, feasible) if not f]
    d_ng = [d for d, f in zip(dis,   feasible) if not f]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- 上段: Di ---
    ax1.plot(betas, dis, color='steelblue', alpha=0.35, lw=1.5)
    if b_ok:
        ax1.scatter(b_ok, d_ok, c='steelblue', s=25, zorder=3, label='実行可能')
    if b_ng:
        ax1.scatter(b_ng, d_ng, c='tomato',    s=25, marker='x', zorder=3, label='制約違反')
    ax1.scatter(
        [result['beta']], [result['induced_drag_N']],
        c='gold', s=220, zorder=6, marker='*', edgecolors='darkorange', linewidths=0.8,
        label=f"最適 β*={result['beta']:.4f},  Di*={result['induced_drag_N']:.4f} N"
    )
    ax1.set_ylabel('誘導抗力 Di [N]', fontsize=11)
    ax1.set_title('β スイープ: 誘導抗力・桁重量の推移', fontsize=12)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.35)

    # --- 下段: W_spar ---
    ax2.plot(betas, wspar, color='seagreen', lw=1.5)
    ax2.scatter(betas, wspar, c=[('seagreen' if f else 'tomato') for f in feasible], s=20, zorder=3)
    ax2.scatter(
        [result['beta']], [result['W_spar_kg']],
        c='gold', s=220, zorder=6, marker='*', edgecolors='darkorange', linewidths=0.8
    )
    ax2.set_xlabel('β（翼根曲げモーメント比）', fontsize=11)
    ax2.set_ylabel('桁重量 W_spar [kg]', fontsize=11)
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[グラフ保存] {save_path}")
    plt.show()


# =============================================================================
# エントリポイント
# =============================================================================

if __name__ == "__main__":
    # opt-ATLAS 条件1（ロングスパン低速）での検証
    params = WingDesignParams(
        W_pilot_kg          = 55.0,
        W_fuselage_kg       = 20.0,
        W_wing_secondary_kg = 8.0,
        span_m              = 32.0,
        V_flight_ms         = 7.2,
        delta_allow_m       = 2.2,
        sigma_allow_MPa     = 300.0,
        mandrel_diameters_mm= np.linspace(120, 45, 8),
        n_spar              = 8,
        n_aero              = 100,
    )

    integrator = DesignIntegratorV3(params)
    result = integrator.optimize(beta_min=0.85, beta_max=1.01, n_beta=100)  # n_betaはβスイープの分割数

    # 積層構成の表示
    print_layup_summary(result)

    # β vs Di グラフの表示・保存
    plot_beta_sweep(result, save_path='beta_sweep_Di.png')
