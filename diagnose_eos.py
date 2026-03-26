"""
EOS サロゲートモデル診断スクリプト
====================================
EOS (XGBoost) モデルの学習データ範囲と予測精度を確認する。

確認事項:
  1. 学習データ（パレートセット）の EI/R 分布範囲
  2. 実際の設計で使う EI 値が学習範囲内かどうか（外挿リスク）
  3. EOS モデルの予測精度確認（物理計算との比較）
  4. snap_optimizer との連携チェック
"""

import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
sys.path.insert(0, '.')

# pickle 復元のために必要な定義（eos_surrogate.py と同一）
def add_physics_features(X):
    log_ei = X[:, 0]
    r = X[:, 1]
    log_r = np.log10(r + 1e-9)
    thickness_index = log_ei - 3 * log_r
    weight_index    = log_ei - 2 * log_r
    return np.column_stack((X, thickness_index, weight_index))

def inverse_log10(x):
    return 10**x

PROJECT_ROOT = Path('.').resolve()
CSV_PATH    = PROJECT_ROOT / "data" / "processed" / "pareto_spar_dataset_eos_model.csv"
MODEL_PATH  = PROJECT_ROOT / "results" / "models" / "spar_weight_surrogate_model_eos_xgb.pkl"

g = 9.80665


# ===========================================================================
# 1. データセット統計
# ===========================================================================

def check_dataset():
    """学習データ（パレートセット）の分布を確認"""
    print("=== 1. データセット統計 ===")

    if not CSV_PATH.exists():
        print(f"  [ERROR] CSV not found: {CSV_PATH}")
        return None

    df = pd.read_csv(CSV_PATH)
    print(f"  データ件数: {len(df):,} 件")
    print(f"  カラム: {df.columns.tolist()}")
    print()

    # EI の範囲（kgf·mm² 単位）
    print(f"  EI [kgf*mm^2] 範囲:")
    print(f"    min  = {df['EI'].min():.4e}")
    print(f"    25%  = {df['EI'].quantile(0.25):.4e}")
    print(f"    50%  = {df['EI'].quantile(0.50):.4e}")
    print(f"    75%  = {df['EI'].quantile(0.75):.4e}")
    print(f"    max  = {df['EI'].max():.4e}")
    print()

    # R（直径）の範囲
    print(f"  R [mm] (直径として使用) 範囲: {df['R'].min():.1f} 〜 {df['R'].max():.1f} mm")
    print()

    # Weight の範囲
    print(f"  Weight [kg/m] 範囲: {df['Weight'].min():.4f} 〜 {df['Weight'].max():.4f} kg/m")
    print()

    return df


# ===========================================================================
# 2. 実際の設計で使う EI 範囲との比較
# ===========================================================================

def check_coverage(df):
    """
    opt-ATLAS 3条件での EI_req が学習データ範囲内か確認する。
    """
    print("=== 2. 実際の設計値との比較（外挿リスク診断）===")

    from src.aerodynamics.aerodynamics_analyzer import AerodynamicsAnalyzer, AircraftAeroParams
    from src.core.ei_estimator import EIEstimator
    from src.structural.structural_analyzer import StructuralAnalyzer

    # opt-ATLAS 3条件のパラメータ
    conditions = [
        {"name": "条件1 (32.5m, 7.2m/s)", "W_kg": 86.0,  "span": 32.5, "V": 7.2,  "delta": 2.2},
        {"name": "条件2 (29m,   8.3m/s)", "W_kg": 83.0,  "span": 29.0, "V": 8.3,  "delta": 2.0},
        {"name": "条件3 (22m,   11m/s)",  "W_kg": 83.06, "span": 22.0, "V": 11.0, "delta": 0.9},
    ]

    EI_max_dataset = df['EI'].max()
    EI_min_dataset = df['EI'].min()

    print(f"  データセット EI 範囲: {EI_min_dataset:.3e} 〜 {EI_max_dataset:.3e} kgf*mm^2")
    print()

    for cond in conditions:
        aero_params = AircraftAeroParams(
            lift_target_N=cond['W_kg'] * g,
            span_m=cond['span'],
            v_flight_ms=cond['V'],
            rho_air=1.154,
            n_segments=100,
        )
        aero = AerodynamicsAnalyzer(aero_params)
        res  = aero.solve(beta=0.99)
        y_m  = res['span_y']

        sa   = StructuralAnalyzer(y_m)
        M_Nm = sa.compute_bending_moment(res['lift_dist_N_m'])

        est    = EIEstimator()
        EI_req = est.calc_EI_req(M_Nm, y_m, cond['delta'])
        # N*mm^2 -> kgf*mm^2
        EI_req_kgf = EI_req / g

        EI_root = EI_req_kgf[0]
        EI_mid  = np.percentile(EI_req_kgf, 50)

        in_range_root = EI_min_dataset <= EI_root <= EI_max_dataset
        in_range_mid  = EI_min_dataset <= EI_mid  <= EI_max_dataset

        status_root = "範囲内" if in_range_root else "外挿!"
        status_mid  = "範囲内" if in_range_mid  else "外挿!"

        # データ範囲のどの位置にあるか（パーセンタイル）
        pct_root = (df['EI'] <= EI_root).mean() * 100
        pct_mid  = (df['EI'] <= EI_mid).mean() * 100

        print(f"  {cond['name']}:")
        print(f"    EI_req 翼根 = {EI_root:.3e} kgf*mm^2  [{status_root}, データの上位 {100-pct_root:.1f}% 付近]")
        print(f"    EI_req 中央 = {EI_mid:.3e} kgf*mm^2   [{status_mid}, データの上位 {100-pct_mid:.1f}% 付近]")
        print()


# ===========================================================================
# 3. EOS モデルの予測精度検証（物理計算との比較）
# ===========================================================================

def check_model_accuracy():
    """
    EOS モデルの予測精度を物理計算（SparCalculator）と比較する。
    """
    print("=== 3. EOS 予測精度の検証（物理計算との比較）===")

    if not MODEL_PATH.exists():
        print(f"  [ERROR] モデルファイルが見つかりません: {MODEL_PATH}")
        print("  eos_surrogate.py を実行してモデルを学習してください")
        return

    from src.core.spar_calculator import SparCalculator
    from src.models.eos_surrogate import add_physics_features

    model = joblib.load(MODEL_PATH)
    calc  = SparCalculator()

    # テストケース: 典型的な翼断面パラメータ
    test_cases = [
        # (直径 mm, ply_counts, 説明)
        (120.0, [1, 2, 6, 1, 0, 0, 0, 0, 1, 1, 1], "翼根 (大径, 多積層)"),
        (90.0,  [1, 2, 4, 1, 0, 0, 0, 0, 1, 1, 1], "翼中間 (中径, 中積層)"),
        (60.0,  [1, 2, 2, 1, 0, 0, 0, 0, 1, 1, 1], "翼中間 (小径, 少積層)"),
        (45.0,  [1, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1], "翼端 (最小径)"),
    ]

    print(f"  {'断面':^18} | {'径':>6} | {'EI [kgf*mm2]':>14} | {'物理重量':>10} | {'EOS予測':>10} | {'誤差':>8}")
    print("  " + "-" * 80)

    for D, ply, name in test_cases:
        EI_kgf, w_phys, t_mm, I_mm4, D_outer = calc.calculate_spec(
            np.array(ply, dtype=float), D
        )
        # EOS 予測（直径 D をそのまま R として使う: SnapOptimizer の実装に合わせる）
        log_ei = np.log10(EI_kgf)
        X_in = np.array([[log_ei, D]])
        w_eos = model.predict(X_in)[0]

        err_pct = abs(w_eos - w_phys) / w_phys * 100
        print(f"  {name:<18} | {D:>5.0f}mm | {EI_kgf:>14.4e} | {w_phys:>9.4f} kg/m | {w_eos:>9.4f} kg/m | {err_pct:>6.2f}%")

    print()


# ===========================================================================
# 4. snap_optimizer の EOS 呼び出しロジック確認
# ===========================================================================

def check_snap_optimizer_eos_interface():
    """
    snap_optimizer 内の EOS 呼び出しで単位・変数名の問題がないか確認する。
    """
    print("=== 4. SnapOptimizer と EOS の連携確認 ===")

    print("  SnapOptimizer 内の EOS 呼び出し:")
    print("    入力: [log10(EI_kgfmm2), R_mm]")
    print("    ただし変数名 'R' は実際は直径 D [mm] を使っている")
    print("    (HANDOVER.md: 'mandrel_R_dist_mm の R は直径として計算')")
    print()

    print("  確認: データセット側でも R は直径か？")
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        # R=120mm のデータを見る
        sample = df[df['R'] == 120].head(3)
        if len(sample) > 0:
            print(f"    R=120mm の重量サンプル: {sample['Weight'].values[:3]}")
            # SparCalculator で D=120mm の計算
            from src.core.spar_calculator import SparCalculator
            calc = SparCalculator()
            ply = np.array([1, 2, 6, 1, 0, 0, 0, 0, 1, 1, 1], dtype=float)
            EI, w, t, _, _ = calc.calculate_spec(ply, 120.0)
            print(f"    物理計算 D=120mm 同等積層の重量: {w:.4f} kg/m")
            print(f"    → 整合性: データの重量と物理計算の重量が近ければ直径扱いと一致")
        print()

    print("  design_integrator_v3.py の単位変換チェーン:")
    print("    EIEstimator 出力  : EI_req [N*mm^2]")
    print("    SnapOptimizer 入力: EI_req / g  [kgf*mm^2]  ← 変換式: / 9.80665")
    print("    SnapOptimizer 出力: EI [kgf*mm^2]")
    print("    StructuralAnalyzer 入力: EI * g  [N*mm^2]  ← 変換式: × 9.80665")
    print()
    print("    design_integrator_v3.py L61: solutions = self.snap_opt.solve(target_ei_dist[i] / self.g)")
    print("    design_integrator_v3.py L85: self.struct.compute_deflection(final_moment, actual_ei * self.g)")
    print("    → 変換方向は正しい")
    print()


# ===========================================================================
# 5. 診断サマリー
# ===========================================================================

def print_summary():
    print("=== 診断サマリー ===")
    print()
    print("  [EOS モデルの主要な既知問題]")
    print()
    print("  問題1: 翼根付近 (EI_req 大) は学習データの上位に集中")
    print("         → 外挿リスクは低いが、上位領域のサンプル密度が低い")
    print("         → 必要ならデータ拡充（大 EI 領域のサンプル増加）")
    print()
    print("  問題2: EOS の役割はあくまで 'SnapOptimizer のナビゲーション'")
    print("         → Phase 2 の物理スナップが最終的な精度を保証")
    print("         → EOS 予測が外れても snap_to_physics が正確な解を返す")
    print()
    print("  問題3: HANDOVER.md 指摘の '変な値' の原因")
    print("         → EI_req が M(y) の形状に比例するため翼端でゼロに近づく")
    print("         → SnapOptimizer が EI≒0 に対して無解または非現実的な解を返す")
    print("         → 対策: EI_req に下限値（最小剛性）を設ける")
    print()
    print("  → Phase B-3 の LayupOptimizer でこの問題に対処予定")
    print()


# ===========================================================================
# メイン
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  EOS サロゲートモデル 診断スクリプト")
    print("=" * 65)
    print()

    df = check_dataset()
    if df is not None:
        check_coverage(df)
        check_model_accuracy()
    check_snap_optimizer_eos_interface()
    print_summary()
