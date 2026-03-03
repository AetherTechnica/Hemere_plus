import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "pareto_spar_dataset_eos_model.csv"
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "spar_weight_surrogate_model_eos_xgb.pkl"
TEST_DATA_PATH = PROJECT_ROOT / "results" / "models" / "test_indices.npy"

# ==========================================
# 特徴量エンジニアリング (他モジュールからインポート可能に)
# ==========================================
def add_physics_features(X):
    """
    X[:, 0]: Log10(EI)
    X[:, 1]: R
    追加特徴量1: Log10(EI) - 3*Log10(R) (肉厚相当)
    追加特徴量2: Log10(EI) - 2*Log10(R) (断面積・重量相当)
    """
    log_ei = X[:, 0]
    r = X[:, 1]
    log_r = np.log10(r + 1e-9)
    
    thickness_idx = log_ei - 3 * log_r
    weight_idx = log_ei - 2 * log_r
    
    return np.column_stack((X, thickness_idx, weight_idx))

def inverse_log10(x):
    return 10**x

# ==========================================
# モデル学習
# ==========================================
def train_eos_model_xgb():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    X = np.column_stack((np.log10(df['EI'].values), df['R'].values))
    y = df['Weight'].values
    
    # データリークを防ぐため、Rに基づいて層化抽出するのが理想だが、
    # 簡略化のため通常スプリットし、インデックスを保存する
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )

    # テストインデックスの保存
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(TEST_DATA_PATH, idx_test)

    # パイプライン構築 (Scaler排除)
    base_pipeline = Pipeline([
        ('feat_eng', FunctionTransformer(add_physics_features)),
        ('xgb', xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            objective='reg:squarederror'
        ))
    ])

    # 【重要】目的変数のLog変換
    model = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log10,
        inverse_func=inverse_log10
    )

    print("Training Eos Model (Log-Target XGBoost)...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"Test R^2   : {model.score(X_test, y_test):.5f}")
    print(f"Test MAPE  : {mape:.3f} %")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_eos_model_xgb()