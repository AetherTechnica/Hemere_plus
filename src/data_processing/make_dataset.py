import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import sys

# プロジェクトルートのパス解決
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.core.spar_calculator import SparCalculator

# ==========================================
# 1. 計算ワーカー関数 (並列処理用)
# ==========================================
def compute_diameter_batch(D, base_options, cap_options):
    calc = SparCalculator()
    batch_data = []
    
    ply_counts = np.zeros(11, dtype=int)
    ply_counts[0] = 1   # Glass
    ply_counts[1] = 2   # Torque
    ply_counts[10] = 1  # Glass

    for base_ply in base_options:
        ply_counts[2] = base_ply
        for cap_config in cap_options:
            ply_counts[3:10] = cap_config
            
            EI, W, t_total = calc.calculate_spec(ply_counts, D)
            
            # 【重要】座屈制約の導入: D/tが大きすぎるペラペラの桁は除外
            if (D / t_total) <= 120.0:
                batch_data.append({
                    "EI": EI,
                    "R": D,
                    "Weight": W,
                    "Thickness": t_total,
                    "PlyConfig": str(ply_counts.tolist())
                })
    return batch_data

# ==========================================
# 2. 全探索データ生成
# ==========================================
def generate_full_search_dataset(n_jobs=-1):
    diameters = np.arange(30.0, 131.0, 1.0) # 解像度を1.0mmに向上
    base_options = range(10)
    cap_options = list(itertools.product(range(3), repeat=7))
    
    print(f"Generating comprehensive dataset (Parallel mode, jobs={n_jobs})...")
    print(f"Diameters: {len(diameters)} steps (30-130mm)")
    
    # 並列処理で一気に計算
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_diameter_batch)(D, base_options, cap_options) 
        for D in tqdm(diameters, desc="Processing Diameters")
    )
    
    # リストの平坦化
    all_data = [item for sublist in results for item in sublist]
    df = pd.DataFrame(all_data)
    print(f"Total Valid Generated Points (D/t <= 120): {len(df):,}")
    return df

# ==========================================
# 3. パレートフィルタリング
# ==========================================
def filter_best_dataset(df, bins=1000):
    print("Filtering Global & Local Pareto Frontier...")
    df['LogEI'] = np.log10(df['EI'])
    df['EI_Bin'] = pd.cut(df['LogEI'], bins=bins)
    
    # 直径と剛性ビンごとの最軽量設計を抽出
    idx = df.groupby(['R', 'EI_Bin'], observed=True)['Weight'].idxmin()
    best_df = df.loc[idx.dropna()].copy().sort_values(['R', 'EI']).reset_index(drop=True)
    
    print(f"Compressed Dataset Size: {len(best_df):,}")
    return best_df

if __name__ == "__main__":
    raw_df = generate_full_search_dataset(n_jobs=-1)
    pareto_df = filter_best_dataset(raw_df, bins=1000)
    
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pareto_spar_dataset_eos_model.csv"
    
    pareto_df.to_csv(csv_path, index=False)
    print(f"Dataset saved to: {csv_path}")