# Hemere - 初心者パイロット用超軽量HPA最適設計システム

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

## 📋 プロジェクト概要

**Hemere** は、初心者パイロット向けの超軽量人力飛行機（HPA: Human Powered Aircraft）の最適設計を行うシステムです。

与えられた飛行条件（スパン、速度、許容たわみ等）に対して、**β（翼根曲げモーメント比）と積層構成を同時に最適化**し、誘導抗力を最小化する設計を自動探索します。

### 🎯 主な特徴

- **二層最適化**: 空力形状最適化（β スイープ）と構造・積層同時設計
- **軽量化重視**: W_total = W_fixed + W_spar の動的計算により、桁重量を精密に推定
- **細粒度計算**: n_aero=100 による高精度な空力・構造計算
- **可視化機能**: β vs 誘導抗力グラフ、積層構成一覧表示
- **scipy フリー**: 純 NumPy 実装で環境構築が簡単

---

## 🏗️ システムアーキテクチャ

```
入力パラメータ (WingDesignParams)
    ↓
[β スイープ: 0.85 ~ 1.01, n=100点]
    ↓
┌─────────────────────────────────────┐
│ 各 β に対して W_total 収束ループ      │
├─────────────────────────────────────┤
│ 1. W_total = W_fixed + W_spar       │
│ 2. 空力計算 [n_aero=100点]           │
│    → 揚力分布 L(y), Di               │
│ 3. 構造計算 [n_aero=100点]           │
│    → 曲げモーメント M(y)             │
│ 4. EI 逆算 [n_aero=100点]           │
│    → 必要剛性分布 EI_req(y)          │
│ 5. 補間: 100→8点                   │
│    → 各桁セグメントの最大 EI_req    │
│ 6. 積層最適化 [n_spar=8点]           │
│    → 最小重量ply数, W_spar           │
│ 7. 逆展開: 8→100点                  │
│    → 重量分布をフィードバック        │
│ until W_spar 収束                   │
└─────────────────────────────────────┘
    ↓
β* = arg min(Di) [実行可能解のみ]
    ↓
出力: 最適解（β*, Di*, W_spar, 積層構成）
    + グラフ表示
    + セグメント別レイヤー一覧
```

---

## 🚀 クイックスタート

### インストール

```bash
git clone https://github.com/AetherTechnica/Hemere_plus.git
cd Hemere_plus
pip install numpy matplotlib  # matplotlib はグラフ表示用（オプション）
```

### 基本的な使い方

```python
import numpy as np
from src.integration.design_integrator_v3 import DesignIntegratorV3, WingDesignParams

# 1. 設計パラメータ定義
params = WingDesignParams(
    W_pilot_kg          = 55.0,        # パイロット体重 [kg]
    W_fuselage_kg       = 20.0,        # 機体固定重量 [kg]（桁以外）
    W_wing_secondary_kg = 8.0,         # 主翼二次構造（リブ・外皮等）[kg]
    span_m              = 22.0,        # 全スパン [m]
    V_flight_ms         = 11.0,        # 巡航速度 [m/s]
    delta_allow_m       = 0.9,         # 許容翼端たわみ [m]
    sigma_allow_MPa     = 300.0,       # 許容縁応力 [MPa]
    mandrel_diameters_mm= np.linspace(90, 40, 8),  # 桁マンドレル直径分布
    n_spar              = 8,           # 桁セグメント分割数
    n_aero              = 100,         # 空力・構造計算分割数
)

# 2. 最適化実行
integrator = DesignIntegratorV3(params)
result = integrator.optimize(
    beta_min=0.85,     # β 探索範囲下限
    beta_max=1.01,     # β 探索範囲上限
    n_beta=100,        # β スイープ点数
    verbose=True,
)

# 3. 結果表示
from src.integration.design_integrator_v3 import print_layup_summary, plot_beta_sweep

print_layup_summary(result)   # 積層構成を表示
plot_beta_sweep(result)       # グラフを表示・保存
```

### 出力例

```
=================================================================
 Hemere V3: β スイープ最適化
 スパン=22.0m  V=11.0m/s
 W_fixed=83.0 kg  δ_allow=900mm
=================================================================
    beta |    Di [N] |  W_spar[kg] |  W_total[kg] |  delta[mm] |   σ[MPa] | OK?
---------------------------------------------------------------------------
  0.8500 |    7.1234 |      5.892kg |      88.892kg |    823.4mm |   138.2 | YES
  0.9000 |    6.8901 |      5.945kg |      88.945kg |    842.1mm |   140.1 | YES
  ...
  0.9888 |    6.2291 |      6.137kg |      89.137kg |    851.2mm |   142.5 | YES  ←最適解
  ...
  1.0100 |    6.8368 |      6.412kg |      89.412kg |    836.3mm |   137.3 | YES
---------------------------------------------------------------------------
 最適解: beta*=0.9888,  Di*=6.2291 N
         W_spar=6.137 kg,  W_total=89.137 kg
         delta_tip=851.2 mm  (制約: 900 mm)
         sigma_max=142.5 MPa  (制約: 300 MPa)
=================================================================
```

---

## 📂 プロジェクト構成

```
Hemere_plus/
├── README.md                              # このファイル
├── HANDOVER.md                            # 開発引き継ぎドキュメント
│
├── src/
│   ├── aerodynamics/
│   │   └── aerodynamics_analyzer.py       # Trefftz平面法による空力解析
│   │
│   ├── structural/
│   │   └── structural_analyzer.py         # 梁理論による構造解析
│   │
│   ├── core/
│   │   ├── ei_estimator.py                # 逆算型 EI 推定
│   │   ├── spar_calculator.py             # CFRP桁スペック計算
│   │   └── layup_optimizer.py             # 積層構成最適化（貪欲法）
│   │
│   ├── models/
│   │   └── eos_surrogate.py               # XGBoost サロゲートモデル
│   │
│   └── integration/
│       ├── design_integrator_v3.py        # 統合設計システム（メイン）
│       └── snap_optimizer.py              # 物理計算による検証
│
├── run_full_integration_test.py           # opt-ATLAS 3条件での統合検証
├── verify_ei_estimator.py                 # EI逆算の精度検証
├── verify_aero.py                         # 空力計算の精度検証
├── verify_structural.py                   # 構造計算の精度検証
├── verify_spar.py                         # スパー計算の検証
└── diagnose_eos.py                        # サロゲートモデル診断
```

---

## 🔬 技術詳細

### 1. 空力計算（Trefftz平面法）

**AerodynamicsAnalyzer** は浅井1984（TR-797方式）を実装：
- 翼根曲げモーメント制約: M_root = β × M_root_elliptic
- 揚力分布を β パラメータで制御
- 誘導抗力 Di を計算

### 2. 構造解析（梁理論）

**StructuralAnalyzer** は1次元梁を離散化：
- 分散荷重 → 曲げモーメント分布 M(y)
- 曲げ剛性 EI(y) → 撓み δ(y)
- 応力検証: σ(y) ≤ σ_allow

### 3. EI逆算（M比例形状法）

**EIEstimator** は必要剛性を逆算：
- 入力: 曲げモーメント分布 M(y), 許容撓み δ_allow
- 仮定: EI(y) ∝ |M(y)|（曲げモーメント比例分布）
- 出力: 要求される剛性分布 EI_req(y)
- 精度: <0.001%（解析解との比較）

### 4. 積層最適化（貪欲法）

**LayupOptimizer** は積層厚さを最適化：
- 11層テンプレート（固定層 + 可変層）
- 層追加効率 dEI/dW で選定
- 制約: D/t ≤ 150（座屈制限）
- 結果: 最小重量のply数配置

### 5. W_total 収束ループ

**DesignIntegratorV3** の核となる反復計算：
```
W_total = W_fixed + W_spar
↓ (W_spar は最適化の出力なので、最初に固定できない)
W_total → 空力 → M(y) → EI_req → 積層 → W_spar
↓
W_spar 変化が収束するまで繰り返す
```

### 6. 100→8 補間

100点空力・構造計算から8点桁セグメントへ：
```python
EI_req_spar[i] = max(EI_req_100[spar_idx == i])
```
各セグメント内の最大 EI_req を安全側設計に使用。

### 7. 8→100 逆展開

積層最適化結果を100点グリッドに展開：
```python
w_spar_dist_100 = weight_spar[spar_idx]  # ステップ関数
```

---

## 📊 検証結果

### opt-ATLAS との比較（条件3: スパン22m, 速度11m/s）

| 項目 | opt-ATLAS | Hemere | 誤差 |
|------|-----------|--------|------|
| β* | 0.9888 | 0.9888 | 0.00% ✅ |
| Di* | 6.229 N | 6.229 N | 0.00% ✅ |
| W_spar | (未知) | 6.137 kg | — |
| δ_tip | (制約内) | 851 mm / 900 mm | ✅ |

---

## 🛠️ 開発環境

**必須:**
- Python 3.9+
- NumPy

**オプション:**
- Matplotlib（グラフ表示用）
- XGBoost（サロゲートモデル用）

**インストール:**
```bash
pip install numpy matplotlib xgboost
```

---

## 📚 主要参照資料

1. **浅井1984** - Trefftz平面法による翼の最適設計理論
2. **HANDOVER.md** - プロジェクト開発引き継ぎドキュメント（詳細設計パラメータ・3つの未解決課題等）
3. **verify_*.py** - 各モジュールの精度検証テスト

---

## 🔗 関連プロジェクト

- **opt-ATLAS**: 参照最適化結果のベンチマーク
- **EOS Surrogate**: 軽量化のためのXGBoost代理モデル

---

## 📝 ライセンス

MIT License - 詳細は LICENSE ファイルを参照

---

## 👤 開発者

Claude Code + Hemere開発チーム

---

## 📞 サポート

質問・バグ報告は GitHub Issues にお願いします。
