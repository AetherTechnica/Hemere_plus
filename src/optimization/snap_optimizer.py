import numpy as np
import pandas as pd
import joblib
import itertools
from pathlib import Path
import sys
import logging

# プロジェクトルートのパス解決
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# 必要なモジュールのインポート
from src.core.spar_calculator import SparCalculator
from src.models.eos_surrogate import add_physics_features, inverse_log10

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =========================================================
# Snap Optimizer クラス (物理スナップ対応版)
# =========================================================
class SnapOptimizer:
    def __init__(self, model_name="spar_weight_surrogate_model_eos_xgb.pkl"):
        """
        AIによる重量推算と物理計算による積層確定を統合するクラス．
        """
        self.calc = SparCalculator()
        self.model_path = PROJECT_ROOT / "results" / "models" / model_name
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Eos Model not found at {self.model_path}. Run eos_surrogate.py first.")
        
        logger.info(f"Loading Eos Engine from {self.model_path}...")
        # add_physics_featuresなどが正しく読み込まれるよう設定
        self.eos_model = joblib.load(self.model_path)
        
        # 探索設定 (デフォルト値)
        self.r_min = 30.0
        self.r_max = 130.0
        self.r_search_step = 0.5
        self.snap_range = 5.0
        self.snap_step = 1.0  # マンドレル刻み
        self.buckling_limit = 150.0  # D/t 制約

    def predict_ideal_spec(self, target_EI):
        """
        Phase 1: Eosモデルによる理想直径(R)の探索
        """
        r_scan = np.arange(self.r_min, self.r_max + self.r_search_step, self.r_search_step)
        
        # 特徴量作成: [Log10(EI), R]
        log_ei = np.log10(target_EI)
        ei_col = np.full_like(r_scan, log_ei)
        X_scan = np.column_stack((ei_col, r_scan))
        
        # 推論実行
        pred_weights = self.eos_model.predict(X_scan)
        
        best_idx = np.argmin(pred_weights)
        return r_scan[best_idx], pred_weights[best_idx]

    def get_feasible_ply_patterns(self):
        """
        製造可能な積層パターンのジェネレータ
        """
        base_options = range(10)
        cap_options = itertools.product(range(3), repeat=7)
        for base_ply, caps in itertools.product(base_options, cap_options):
            ply_counts = np.zeros(11, dtype=int)
            ply_counts[0] = 1   # Glass (In)
            ply_counts[1] = 2   # Torque
            ply_counts[2] = base_ply
            ply_counts[3:10] = caps
            ply_counts[10] = 1  # Glass (Out)
            yield ply_counts

    def snap_to_physics(self, target_EI, ideal_r, top_n=3):
        """
        Phase 2: 物理スナップ
        要求剛性と座屈制約を満たす設計を理想直径の周辺で全探索し，上位N件を返す．
        """
        r_start = max(self.r_min, np.floor(ideal_r - self.snap_range))
        r_end = min(self.r_max, np.ceil(ideal_r + self.snap_range))
        candidate_diameters = np.arange(r_start, r_end + self.snap_step, self.snap_step)
        
        feasible_solutions = []
        
        for D in candidate_diameters:
            for ply_counts in self.get_feasible_ply_patterns():
                # 物理スペック計算
                real_EI, real_W, t_total, real_I_mm4, real_D_outer = \
                    self.calc.calculate_spec(ply_counts, D)

                # 1. 剛性制約チェック
                if real_EI < target_EI:
                    continue

                # 2. 座屈制約チェック (D/t)
                if (D / t_total) > self.buckling_limit:
                    continue

                # 解の保存（I_mm4, D_outer_mm を追加）
                feasible_solutions.append({
                    "Diameter": D,
                    "Weight": real_W,
                    "EI": real_EI,
                    "Thickness": t_total,
                    "D_t": D / t_total,
                    "Ply_Config": ply_counts.tolist(),
                    "Margin_Pct": (real_EI - target_EI) / target_EI * 100,
                    "I_mm4": real_I_mm4,
                    "D_outer_mm": real_D_outer,
                })
        
        # 重量の昇順でソートして上位を返す
        sorted_sol = sorted(feasible_solutions, key=lambda x: x["Weight"])
        return sorted_sol[:top_n]

    def solve(self, target_EI):
        """
        メイン最適化フロー: AIナビゲーション -> 物理スナップ確定
        """
        logger.info(f"Solving for Target EI: {target_EI:.2e} Nmm^2")
        
        # Step 1: AIによるナビゲーション
        ideal_r, ideal_w = self.predict_ideal_spec(target_EI)
        logger.info(f"AI Guide -> Ideal Diameter: {ideal_r:.1f} mm, Est. Weight: {ideal_w:.4f} kg/m")
        
        # Step 2: 物理スナップ
        solutions = self.snap_to_physics(target_EI, ideal_r)
        
        if not solutions:
            logger.error("No feasible solution found with current constraints.")
            return None
        
        # 結果の出力
        best = solutions[0]
        logger.info("✅ Best Physical Solution Found:")
        logger.info(f"   - Diameter  : {best['Diameter']:.1f} mm")
        logger.info(f"   - Weight    : {best['Weight']:.4f} kg/m")
        logger.info(f"   - Margin    : +{best['Margin_Pct']:.2f} %")
        logger.info(f"   - D/t Ratio : {best['D_t']:.1f} (Limit: {self.buckling_limit})")
        logger.info(f"   - Ply Config: {best['Ply_Config']}")
        
        return solutions

# =========================================================
# 実行部
# =========================================================
if __name__ == "__main__":
    opt = SnapOptimizer()
    
    # テスト用剛性 (5e10 Nmm^2)
    test_ei = 4.0e9
    results = opt.solve(test_ei)
    
    if results:
        # 他の候補も表示してみる
        print("\n--- Alternative Solutions ---")
        for i, res in enumerate(results[1:], 2):
            print(f"Rank {i}: R={res['Diameter']:.1f}, W={res['Weight']:.4f}, Margin={res['Margin_Pct']:.1f}%")