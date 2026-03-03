import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# 特徴量関数のインポート (Pickleのロードに必須)
from src.models.eos_surrogate import add_physics_features, inverse_log10

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "pareto_spar_dataset_eos_model.csv"
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "spar_weight_surrogate_model_eos_xgb.pkl"
TEST_DATA_PATH = PROJECT_ROOT / "results" / "models" / "test_indices.npy"
SAVE_IMG_PATH = PROJECT_ROOT / "results" / "figures" / "eos_model_evaluation.png"

def evaluate_eos_performance():
    if not (CSV_PATH.exists() and MODEL_PATH.exists() and TEST_DATA_PATH.exists()):
        print("Error: Required files not found. Run training script first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    model = joblib.load(MODEL_PATH)
    test_indices = np.load(TEST_DATA_PATH)

    X_all = np.column_stack((np.log10(df['EI'].values), df['R'].values))
    y_all = df['Weight'].values

    # 正しいテストデータの復元
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    
    y_pred = model.predict(X_test)
    
    rel_error = (y_pred - y_test) / y_test * 100.0
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    log_EI_test = X_test[:, 0]
    R_test = X_test[:, 1]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Eos Model Evaluation Dashboard\nR2={r2_score(y_test, y_pred):.4f}, MAPE={mape:.3f}%", fontsize=16)

    # 1. Accuracy Check (Log-Log Scale)
    ax1 = fig.add_subplot(2, 3, 1)
    sc1 = ax1.scatter(y_test, y_pred, c=np.abs(rel_error), cmap='coolwarm', s=10, alpha=0.7, vmin=0, vmax=np.percentile(np.abs(rel_error), 95))
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax1.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('Actual Weight [kg/m]'); ax1.set_ylabel('Predicted Weight [kg/m]')
    ax1.set_title('Accuracy Check (Log-Log)')
    plt.colorbar(sc1, ax=ax1, label='Abs Rel Error (%)')
    ax1.grid(True, alpha=0.3)

    # 2. Residual vs Predicted
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(y_pred, rel_error, alpha=0.5, s=10, color='teal')
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xscale('log')
    ax2.set_xlabel('Predicted Weight [kg/m]')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)

    # 3. Feature Importance
    ax3 = fig.add_subplot(2, 3, 3)
    xgb_model = model.regressor_.named_steps['xgb']
    importances = xgb_model.feature_importances_
    features = ['Log10(EI)', 'R', 'Thickness_Idx', 'Weight_Idx']
    ax3.bar(features, importances, color='coral')
    ax3.set_title('XGBoost Feature Importances')
    ax3.tick_params(axis='x', rotation=15)

    # 4. Error Histogram
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(rel_error, bins=50, color='teal', edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='red', linestyle='--')
    ax4.set_xlabel('Relative Error (%)')
    ax4.set_title(f'Error Distribution\nMean: {np.mean(rel_error):.2f}%')
    ax4.grid(True, alpha=0.3)

    # 5. Bias vs Diameter
    ax5 = fig.add_subplot(2, 3, 5)
    sc5 = ax5.scatter(R_test, rel_error, c=log_EI_test, cmap='viridis', s=10, alpha=0.7)
    ax5.axhline(0, color='red', linestyle='--')
    ax5.set_xlabel('Diameter R [mm]')
    ax5.set_ylabel('Relative Error (%)')
    ax5.set_title('Bias vs Diameter')
    plt.colorbar(sc5, ax=ax5, label='Log10(EI)')
    ax5.grid(True, alpha=0.3)

    # 6. Bias vs Stiffness
    ax6 = fig.add_subplot(2, 3, 6)
    sc6 = ax6.scatter(log_EI_test, rel_error, c=R_test, cmap='plasma', s=10, alpha=0.7)
    ax6.axhline(0, color='red', linestyle='--')
    ax6.set_xlabel('Log10(EI)')
    ax6.set_ylabel('Relative Error (%)')
    ax6.set_title('Bias vs Stiffness')
    plt.colorbar(sc6, ax=ax6, label='R [mm]')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    SAVE_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_IMG_PATH, dpi=300)
    print(f"Evaluation plot saved to: {SAVE_IMG_PATH}")
    # plt.show()

if __name__ == "__main__":
    evaluate_eos_performance()