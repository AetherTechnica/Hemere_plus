# Hemere 開発引き継ぎ資料

**作成日**: 2026-03-26
**作成者**: SH!N（WASA OB）× Claude Code
**目的**: 別セッションのClaude Codeへの開発引き継ぎ

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [背景：opt-ATLAS とは](#2-背景opt-atlas-とは)
3. [Hemere の位置づけ](#3-hemere-の位置づけ)
4. [現在のHemereコード構成](#4-現在のhemereコード構成)
5. [問題の核心：3つの未解決課題](#5-問題の核心3つの未解決課題)
6. [採用するアーキテクチャ案](#6-採用するアーキテクチャ案)
7. [opt-ATLAS の物理ロジック詳細](#7-opt-atlas-の物理ロジック詳細)
8. [集中積層のEI計算](#8-集中積層のei計算)
9. [設計パラメータ・定数](#9-設計パラメータ定数)
10. [次の開発ステップ](#10-次の開発ステップ)
11. [注意事項・既知バグ](#11-注意事項既知バグ)


---

## 1. プロジェクト概要

### 何を作りたいか

**翼スパンと飛行速度 V を入力したとき、誘導抗力 Di を最小化する最適な循環分布パラメータ β と桁積層構成を求める設計最適化ツール。**

- 入力: スパン長、飛行速度、機体重量（桁以外）、マンドレル径分布、制約値
- 出力: 最適 β、最適積層構成（各セグメントの層数分布）、誘導抗力 Di

### ユーザー・利用場面

- 開発者: WASAという鳥人間プロジェクト学生サークルのOB（SH!N）
- 将来的には後輩設計者への設計支援ツールとして共有
- 何度もパラメータを変えて回すことが想定される（比較検討用途）
- 現時点は「ロジックの完成」を優先。将来的に統合設計プログラムに組み込む予定

---

## 2. 背景：opt-ATLAS とは

### TR-797法（浅井1984）

人力飛行機の主翼設計における理論的基盤。

- 翼根曲げモーメントを楕円分布の **β 倍**に制約したときの最小誘導抗力循環分布を解析的に求める手法
- β=1: 楕円循環分布（誘導抗力係数は最小だが翼根荷重が最大）
- β<1: 翼根寄せ（翼根荷重を減らせるが誘導抗力係数は悪化）
- **キモ**: β を下げて翼根荷重を減らすと桁を軽くできる → 機体重量 W が減る → Di = W×Cdi/CL が下がりうる

### opt-ATLAS が解いた問題

```
minimize  Di(β, n_0deg_dist)
subject to  σ_max ≤ σ_allow
            δ_tip ≤ δ_allow
```

**設計変数**:
- β ∈ [0.80, 1.01]（実数）
- n_0deg[j] ∈ [1, 20]（整数）× M セグメント

**積層構成（全周積層のみ）**:
```
[90°全周 × 1, 45°全周 × 1, 45°全周 × 1, 0°全周 × n_0deg, 90°全周 × 1]
```
→ 全層が全周（360°）積層なので断面は常に同心円環 → `I = (π/64)(D_out⁴ - D_in⁴)` で計算可能

**最適化手法**: MATLABのGA（遺伝的アルゴリズム）、PopulationSize=1000, MaxGenerations=500

**実績**:
| 条件 | スパン | 速度 | 最適β | Di |
|------|--------|------|-------|----|
| 低速ロングスパン | 32.5m | 7.2m/s | 0.9794 | 10.301N |
| 中速中スパン | 29m | 8.3m/s | 0.9809 | 7.670N |
| 高速ショートスパン | 22m | 11m/s | 0.9888 | 6.229N |

→ いずれも β ≈ 0.98〜0.99 でわずかに翼根寄せが最適

### opt-ATLAS の実装（MATLAB: opt_ATLAS.m）

このファイル（`C:\Users\Shidw\Downloads\opt_ATLAS (1).m`）が「正しい物理ロジック」の基準。詳細は [Section 7](#7-opt-atlas-の物理ロジック詳細) 参照。

---

## 3. Hemere の位置づけ

### 集中積層とは

WASA で実際に使われる桁積層方式。全周に貼るのではなく、**桁の上下（曲げ応力が高い部分）に集中して**プリプレグを貼る。

```
全周積層 (opt-ATLAS):         集中積層 (Hemere):
  ┌─────────┐                    ┌─────────┐
  │█████████│ ← 全周に積層        │    █    │ ← 上部のみ積層
  │         │                    │         │
  │█████████│                    │    █    │ ← 下部のみ積層
  └─────────┘                    └─────────┘
```

積層角度（カバー角度）の例:
- 90°: 全周（360°）
- 50°: 上下それぞれ頂点から±50°の範囲（計100°）
- 20°: 上下それぞれ頂点から±20°の範囲（計40°）

2ply毎に5°ずつずらして積層（製造上の制約）。最大16層程度（前後剛性を考慮）。

### なぜ難しいか

```
全周積層の設計変数: [β, n_0deg[0], n_0deg[1], ..., n_0deg[M-1]]
  → M+1 次元（M=8〜11程度）→ GAで十分探索できる

集中積層の設計変数: [β, ply_angle[0]×n_ply[0], ply_angle[1]×n_ply[1], ..., × M セグメント]
  → 1セグメントでも (角度パターン) × (各層枚数) の組み合わせで数万通り
  → M セグメントでは M乗で組み合わせ爆発 → 次元の呪い
```

### surrogateモデルによるアプローチ（現在の実装方針）

「積層変数を陽に扱わず、桁重量と剛性の相関を利用する」という方針：

1. **EOS（Eos Surrogate）モデル**: `[EI, 径] → 重量` を XGBoost で近似
2. **2段階最適化**:
   - Phase 1（AI推論）: EI_req → 理想的な径を AI が高速予測
   - Phase 2（物理スナップ）: 理想径周辺で物理的に正確な積層構成を探索

---

## 4. 現在のHemereコード構成

### ディレクトリ

```
C:\Users\Shidw\HPA\Hemere\
├── src/
│   ├── core/
│   │   └── spar_calculator.py          # 集中積層 EI・重量計算
│   ├── aerodynamics/
│   │   └── aerodynamics_analyzer.py    # TR-797法 空力解析
│   ├── structural/
│   │   └── structural_analyzer.py      # 梁曲げ解析（M, δ, σ）
│   ├── optimization/
│   │   └── snap_optimizer.py           # 2段階最適化（AI+物理スナップ）
│   ├── models/
│   │   ├── eos_surrogate.py            # EOS XGBoost モデル学習
│   │   └── evaluate_eos.py             # モデル性能評価
│   ├── data_processing/
│   │   └── make_dataset.py             # パレート最適データセット生成
│   └── integration/
│       ├── design_integrator.py        # 統合設計フロー v2
│       ├── design_integrator_v3.py     # 統合設計フロー v3
│       ├── final_design_generator.py   # 固定直径設計（本運用版）
│       └── stiffness_searcher.py       # 剛性分布最適化（SLSQP）
├── analyze_spar_error.py               # 設計空間分析
└── README.md
```

### 各モジュールの役割と主要関数

#### `spar_calculator.py`
```python
calculate_spec(ply_counts: list[11], diameter_mm: float)
  -> (EI_kgfmm2: float, weight_kg_per_m: float, thickness_mm: float)
```
- 積層テンプレート: index 0〜10 の11層
- 全周積層層（index 0,1,2,10）と集中積層層（index 3〜9）が混在
- 検証済み: D=120mm, ply=[1,4,6,1,0,0,0,0,1,1,1] → EI = 1.4363×10¹⁰ kgf·mm²

#### `aerodynamics_analyzer.py`
```python
AerodynamicsAnalyzer(params: AircraftAeroParams).solve(beta: float)
  -> dict{induced_drag_N, gamma, lift_dist_N_m, span_y, efficiency}
```
- TR-797法（制限渦格子法）を実装
- opt-ATLASの `calculateAerodynamics` に相当

#### `structural_analyzer.py`
```python
StructuralAnalyzer.compute_bending_moment(net_load, y) -> moment
StructuralAnalyzer.compute_deflection(moment, ei_dist, y) -> (deflection, slope)
StructuralAnalyzer.check_strength(moment, I_dist, D_outer_dist) -> sigma_max_Pa
```

#### `snap_optimizer.py`
```python
SnapOptimizer.solve(target_EI) -> solutions[top_n]
  # Phase 1: predict_ideal_spec(EI) → AI で理想径を予測
  # Phase 2: snap_to_physics(EI, ideal_r) → 物理的に正確な積層探索
```

#### `stiffness_searcher.py`
```python
StiffnessSearcherV2.optimize_ei_distribution(initial_beta) -> ei_dist [Nmm²]
```
- SLSQPで EI 分布を最適化（β固定時）

### 現在の計算フロー

```
β
 │
 ▼
AerodynamicsAnalyzer.solve(β)
  → 揚力分布 L(y)
 │
 ▼
StructuralAnalyzer.compute_bending_moment(L(y) - 自重)
  → 曲げモーメント M(y)
 │
 ▼
[未確立] EI_req(y) の逆算
  ← たわみ制約 δ_tip ≤ δ_allow から
 │
 ▼
SnapOptimizer.solve(EI_req[j]) for j in segments
  → Phase 1: EOS モデル [EI, R] → W_pred
  → Phase 2: 物理スナップ → 積層構成
 │
 ▼
w_struct(y) → W_struct → W_total
  ← 重量収束ループ（上に戻る）
 │
 ▼（収束）
最適 β の探索（βスイープ or GA）
```

---

## 5. 問題の核心：3つの未解決課題

### ① EI_req の逆算式が未確立

**問題**: たわみ制約 `δ_tip ≤ δ_allow` から、各位置の必要剛性 `EI_req(y)` を求める式がない。

物理的には:
```
κ(y) = M(y) / EI(y)          （曲率）
θ(y) = ∫₀ʸ κ(y') dy'         （傾斜角）
δ(y) = ∫₀ʸ θ(y') dy'         （たわみ）
δ_tip = δ(L/2) ≤ δ_allow
```

これは `EI(y)` の分布に対する積分不等式であり、`EI(y)` を一意に逆算することは一般に不可能。
何らかの近似・仮定が必要（例: EIが一定、あるいはM(y)に比例するなど）。

**現状**: `stiffness_searcher.py` では SLSQP を使って `EI_dist` を数値最適化しているが、
この逆算ロジック自体の物理的な正しさが未検証。

### ② EI → 重量 surrogate モデルが怪しい

**問題**: EOS モデル（XGBoost）の回帰精度（R²≈0.95）は良好だが、
統合フローで使うと「変な値」が出る。

**疑われる原因**:
- ① の EI_req 計算が間違っているため、モデルに渡すEI値が物理的に非現実的
- 学習データ（パレートセット）の範囲外の入力になっている
- 特徴量エンジニアリング（`log10(EI)`, `R`）の定義が実用時と学習時でずれている

**EOSモデルの仕様**:
```python
入力: [log10(EI_kgfmm2), R_mm]  ← 単位に注意
出力: weight_kg_per_m
学習データ: pareto_spar_dataset_eos_model.csv（約50,000点）
モデル: XGBoost（1500 trees, depth=6）
```

### ③ EI → 積層構成の逆算方法が未確立

**問題**: `EI_req[j]` と `径[j]` から「その EI を満たす最小重量の積層構成」を求める方法が未定。

**採用予定の方針（ユーザー提案）**:
- 解析的逆算（二分法など）で最小枚数の積層構成を求める
- 実際に製造可能かのチェック（D/t 比による座屈制約など）をここで行う

---

## 6. 採用するアーキテクチャ案

ユーザー提案の「2フェーズ分離」アプローチ。
積層変数を最適化ループから切り離すことで次元の呪いを回避する。

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【フェーズ 1: 空力-構造連成最適化】
設計変数: β のみ（1次元）

  β
  │
  ▼
[Trefftz/TR-797]
  → 循環分布 Γ(y)
  → 揚力分布 L(y)
  │
  ▼ （W_total を仮定して開始）
[曲げモーメント計算]
  → M(y) = ∫_y^(L/2) [L(y') - w(y')] × (y'-y) dy'
  │
  ▼
[EI_req 逆算] ← ★課題①
  EI_req(y): δ_tip ≤ δ_allow を満たす最小の EI 分布
  │
  ▼
[EI → 重量 surrogate] ← ★課題②
  w(y) = EOS(EI_req(y), 径(y))
  │
  ▼
  W_struct = 2 × ∫ w(y) dy
  W_total = W_fixed + W_struct
  │
  ← 収束？ → No: 上に戻る
  ↓ Yes
  Di を記録

βをスイープして Di が最小の β* を選択

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【フェーズ 2: 積層構成逆算】
設計変数: なし（EI_req が与えられた後の後処理）

  EI_req[j], 径[j]
  │
  ▼
[最小重量積層探索] ← ★課題③
  二分法 or 貪欲法で積層構成を決定
  │
  ▼
[製造可能性チェック]
  D/t 比 ≤ 座屈制約
  実際に製作可能な積層数か
  │
  ▼
最終設計仕様（積層構成、重量、EI 分布）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 7. opt-ATLAS の物理ロジック詳細

**基準ファイル**: `C:\Users\Shidw\Downloads\opt_ATLAS (1).m`
**単位系**: kgf・mm 系（Pythonへ移植時は要変換）

### 7.1 calculateStiffness（全周積層EI計算）

```matlab
function [EI_kgfmm2, Area_mm2, I_total_mm4, D_outer_mm] = calculateStiffness(ply_number, pp, R_mm)
% ply_number: 各層の積層数 [1, 1, 1, n_0deg, 1] 等
% pp:         各層の繊維角度 [90, 45, 45, 0, 90] 等
% R_mm:       マンドレル半径 [mm]（注: 半径！）

PLY_THICKNESS_MM = 0.111;
material = [0 13000 1900 900 22000 1900 800];  % E値 [kgf/mm²]
% material(4)=E90=900, material(5)=E0=22000, material(6)=E45=1900

% 厚み計算
thickness_mm = ply_number * PLY_THICKNESS_MM;

% 内径・外径（全周積層なので両側に積層 → ×2）
inner_mm(1) = R_mm;
inner_mm(i) = inner_mm(i-1) + 2 * thickness_mm(i-1);  % ← ×2 に注意
outer_mm    = inner_mm + 2 * thickness_mm;             % ← ×2 に注意

% 断面二次モーメント（完全な円環）
I_mm4 = (pi/64) * (outer_mm^4 - inner_mm^4);

% EI の積み上げ
EI = sum(I_mm4 * E_layer)
```

**重要**: `inner_mm` の増分が `2 * thickness`（両側積層）のため、
全周積層では ply_count=n のとき片側厚みは `n * PLY_THICKNESS`、直径増分は `2n * PLY_THICKNESS`。

### 7.2 calculateAerodynamics（Trefftz/TR-797法）

```matlab
% 連立方程式: [A+A', -c, -b; -c', 0, 0; -b', 0, 0] × [γ; μ1; μ2] = [0; -1; -β]
% → β を右辺に与えることで翼根曲げモーメントを制約
% 解から循環分布 g を求め、物理量に変換:
Gamma = (Lift / (2*le*rho*Uinf)) * g;   % [m²/s]
L_dist = rho * Uinf * Gamma;            % [N/m]
Di = efficiency_inverse * Di_ellipse;   % [N]
```

### 7.3 core_evaluation（計算フロー）

```
1. n_plies → calculateStiffness → EI(y), Area(y), I(y), D_outer(y)
2. Area(y) × density → W_struct_kg → W_total_N
3. createAircraft + calculateAerodynamics(β) → Di, L(y)
4. L(y) - w_struct(y) - w_fixed = Net_Load(y)
5. M(y) = ∫_y^end Net_Load × (y'-y) dy'  （翼端から積分）
6. M → kgf·mm 変換
7. 制約①: σ = M/Z ≤ σ_allow  （Z = I / (D_outer/2)）
8. 制約②: δ_tip = ∫∫ (M/EI) dy dy ≤ δ_allow
```

### 7.4 単位系まとめ

| 量 | opt-ATLAS単位 | Python変換 |
|----|--------------|-----------|
| EI | kgf·mm² | × 9.80665 → N·mm² |
| M（モーメント）| kgf·mm | × 9.80665 → N·mm |
| 応力 | kgf/mm² | × 9.80665 → N/mm² = MPa |
| 重量 | kg | × 9.80665 → N |
| 長さ | mm | ÷ 1000 → m |

---

## 8. 集中積層のEI計算

### spar_calculator.py の積層テンプレート（11層）

```
index:      0    1    2    3    4    5    6    7    8    9   10
カバー角:  90°  90°  90°  50°  45°  40°  35°  30°  25°  20°  90°
繊維角:    90°  45°   0°   0°   0°   0°   0°   0°   0°   0°  90°
材料:      24t  40t  40t  40t  40t  40t  40t  40t  40t  40t  24t
E[kgf/mm²]:900 1900 22000 22000 22000 22000 22000 22000 22000 22000 900
役割:     保持 捻り  ←──────────── 曲げ担当 ────────────→  保持
固定枚数:   1    4    1    ←── 設計変数 n_cap[j] ──→      1
```

### EI計算式（カバー角度による補正）

```python
# opt-ATLAS（全周積層）:
I = (pi/64) * (D_outer^4 - D_inner^4)         # 完全な円環

# Hemere（集中積層）:
Ix_factor = (angle*pi/360 + sin(2*angle*pi/180)/4) / 16
I_partial = (D_outer^4 - D_inner^4) * Ix_factor
```

カバー角 90°（全周）のとき:
```
Ix_factor = (90°×π/360 + sin(180°)/4) / 16
          = (π/4 + 0) / 16
          = π/64  ← 完全な円環の係数と一致 ✓
```

### 検証済み計算値

D=120mm（内径）, ply=[1, 4, 6, 1, 0, 0, 0, 0, 1, 1, 1]:
- **EI = 1.4363×10¹⁰ kgf·mm²**（エクセル「主翼設計2025_ver8.xlsx」の桁構成シートと完全一致）

---

## 9. 設計パラメータ・定数

### WASAの典型的なパラメータ

```python
# 機体パラメータ（WASA 27代想定）
W_pilot_kg       = 58       # パイロット体重
W_fixed_other_kg = 77       # パイロット(58) + 機体構造(19) [kg]
W_wing_fixed_kg  = 9.0      # リブ・外皮等 [kg]
span_m           = 32.0     # 全スパン [m]（半スパン = 16m）
V_flight         = 7.2      # 飛行速度 [m/s]
rho_air          = 1.154    # 空気密度 [kg/m³]
delta_allow_m    = 2.2      # 許容翼端たわみ [m]
sigma_allow_Pa   = 300e6    # 許容曲げ応力 [Pa]
g                = 9.80665  # 重力加速度 [m/s²]
```

### プリプレグ材料定数

```python
# 厚み（1ply あたり）
t_ply_24t = 0.125   # mm/ply（ガラス）
t_ply_40t = 0.111   # mm/ply（カーボン）

# ヤング率（繊維方向）
E_24t_0deg  = 13000  # kgf/mm²
E_24t_45deg = 1900   # kgf/mm²
E_24t_90deg = 900    # kgf/mm²
E_40t_0deg  = 22000  # kgf/mm²
E_40t_45deg = 1900   # kgf/mm²
E_40t_90deg = 800    # kgf/mm²

# CFRP密度
rho_cfrp = 1559  # kg/m³
```

### マンドレル径について

- **製造元**: スリーホープ（FRPパイプ）
- 規格品の内径一覧がある（φ3〜φ300mm など多数）
- **将来的には**: スリーホープの規格径リストから選択する仕組みにする
- **現在は**: `np.linspace(内径最大, 内径最小, M_segments)` の連続近似で代用

```python
# 現在の近似値（WASAの典型設計）
mandrel_diameters_mm = np.linspace(120, 45, 8)  # 8セグメント
# 注意: これは「直径」[mm]（opt-ATLASの R_mm は「半径」と異なる）
```

---

## 10. 次の開発ステップと現在の進捗

**方針**: コードを書く前に物理モデルを確立する（B方針）。
surrogateモデルや最適化の前に、基盤となる物理計算の正しさを担保する。

### ✅ 完了済み（2026-03-26）

#### Step 1: 物理モデル検証 - spar_calculator
- `calculate_spec()` に返値追加: `I_mm4`, `D_outer_mm`
- 検証: D=120mm, ply=[1,4,6,1,0,0,0,0,1,1,1] → EI=1.4363×10¹⁰ kgf·mm²（エクセル値と 誤差0.003%）

#### Step 2: 物理モデル検証 - aerodynamics_analyzer
- TR-797法実装の正確性確認
- opt-ATLAS の `calculateAerodynamics` と完全一致（差0.000%）

#### Step 3: 物理モデル検証 - structural_analyzer
- 曲げモーメント・たわみ計算の検証
- 応力計算の修正: E固定の不正確さを廃止 → I_mm4, D_outer_mm を直接受け取り
- scipy 依存を除去（`_cumtrapz` 純NumPy実装）
- opt-ATLAS と完全一致（差0.000%）

#### Step 4-1: バグ修正（4ファイル）
1. ✅ `spar_calculator.py`: API 返値追加
2. ✅ `structural_analyzer.py`: scipy 除去、check_strength API 変更
3. ✅ `snap_optimizer.py`: calculate_spec() 新返値に対応
4. ✅ `design_integrator_v3.py`: キー名不整合修正、solutions[0] 追加、check_strength 新API、単位変換（N·mm² ↔ kgf·mm²）

#### Step 4-2: ei_estimator.py の新規実装
- **ファイル**: `C:\Users\Shidw\HPA\Hemere\src\core\ei_estimator.py`
- **実装内容**:
  ```python
  class EIEstimator:
      def calc_EI_req(self, M_Nm: np.ndarray, y_m: np.ndarray, delta_allow_m: float) -> np.ndarray:
          """
          M比例形状法 + 二分法で EI_req を逆算

          アルゴリズム:
          1. EI_shape(y) = |M(y)| / max(|M(y)|)  （正規化されたモーメント形状）
          2. EI_req(y) = α × EI_shape(y)  （EI がモーメントに比例と仮定）
          3. 二分法で α を決定: compute_deflection(M, α × EI_shape) の翼端たわみが delta_allow になるように
          4. 返却: EI_req_Nmm2 [N·mm²]

          Returns:
              np.ndarray: EI_req の分布 [N·mm²]
          """
  ```

### ⏳ 進行中 / 保留

#### Phase B-1: EI_req 逆算の物理式を確立

**目標**: `M(y)` と `δ_allow` から `EI_req(y)` を求める式を決める。

**候補アプローチ**:

**(a) 比例配分法**（シンプル・近似）
```
EI_req(y) = M(y) / κ_allow

ここで κ_allow は δ_tip = δ_allow を満たす一様曲率（簡略近似）
δ = κ × L²/2 → κ_allow = 2×δ_allow / (L/2)²
```

**(b) 形状保存法**（物理的に自然）
```
EI_req(y) = α × M(y)  （EIがモーメントに比例と仮定）
αを δ_tip = δ_allow で決定
```

**(c) 数値逆問題**（最も正確だが複雑）
```
EI_dist を自由変数として、δ_tip(EI_dist) = δ_allow を満たす
最小重量の EI_dist を SLSQP等で求める（現在の stiffness_searcher.py）
```

→ **opt-ATLAS との整合性確認**: opt-ATLAS では EI が先に決まってから δ を計算する
（順方向）。Hemere での逆問題は opt-ATLAS にはない概念のため、慎重に設計する。

### Phase B-1-1: M比例形状法の検証テスト

設計後、下記の検証スクリプトで動作確認:

```bash
cd C:\Users\Shidw\HPA\Hemere
/c/Users/Shidw/anaconda3/python.exe -c "
import numpy as np, sys; sys.path.insert(0,'.')
from src.core.ei_estimator import EIEstimator
from src.structural.structural_analyzer import StructuralAnalyzer

# テストケース: 単純な集中荷重下での梁
y = np.linspace(0, 16, 200)
M = 0.5 * 100 * (16 - y)**2  # 集中荷重のモーメント分布

est = EIEstimator()
EI_req = est.calc_EI_req(M, y, 0.8192)

sa = StructuralAnalyzer(y)
d, _ = sa.compute_deflection(M, EI_req)

print('delta_tip =', d[-1]*1000, 'mm (期待値: 819.2 mm)')
print('EI_req min/max =', np.min(EI_req), '/', np.max(EI_req), '[N*mm^2]')
"
```

### Phase B-2: surrogate モデルの問題特定

**確認事項**:
1. EOS モデルの入力単位: `EI [kgf·mm²]` か `[N·mm²]` か → `make_dataset.py` で確認
2. 外挿の有無: 実際に使う EI 値が学習データ範囲内か確認
3. B-1 で EI_req が正しく計算されてから再検証

### Phase B-3: EI → 積層構成逆算の実装

```python
def find_min_weight_layup(EI_req_kgfmm2: float, diameter_mm: float) -> dict:
    """
    EI_req を満たす最小重量の積層構成を二分法で求める。

    アルゴリズム（貪欲法案）:
    1. 固定層（index 0,1,2,10）を初期値として EI_base を計算
    2. EI_base < EI_req の間、EI/重量効率が最大の層を1枚追加
       ※ カバー角が大きい層ほど EI 効率が高い（index 3=50° が最効率）
    3. 最終的な積層構成を返す

    チェック項目:
    - D_outer / t_total ≤ 座屈制約（150程度）
    - 積層数が現実的範囲内（≤ 16 layer）
    """
```

### Phase B-4: 統合テストと検証

opt-ATLAS と同じ条件で解いて結果を比較：

```python
# opt-ATLAS 条件3 での検証（スパン22m, 速度11m/s）
params_atlas = {
    'W_fixed_other_kg': 20+55,
    'W_wing_fixed_kg': 8,
    'span_m': 22.0,  # ← 全スパン（条件3）
    'V_flight': 11,
    'delta_allow_m': 0.9,
    # ⚠️【検証済み】opt-ATLAS の mandrel_R_dist_mm は変数名が R（半径）だが、
    #    コード内部では「直径」として計算している。
    #    Hemere の diameter_mm と同じ解釈（直径）で使える。
    'mandrel_diameters_mm': np.linspace(90, 40, 8),  # 直径 [mm]
    'M_segments': 8,
}
# 期待値: Di ≈ 6.229N, β ≈ 0.9888, 主翼重量 ≈ 13.06kg
```

---

## 11. 注意事項・既知バグ

### 単位・変数名のトラップ

| 注意点 | 詳細 |
|--------|------|
| `span_m` は **全スパン** | Trefftz Solver には `span_m/2` を渡す。`span_m=16` とすると全infeasible になるバグが過去に発生 |
| `mandrel_R_dist_mm` の "R" | **【検証済み】** opt-ATLASも Hemere も実際は**直径**として計算している。変数名が R（半径）でも内部では直径として使われており、両者は整合している。 |
| EI の単位 | spar_calculator は `[kgf·mm²]`、構造計算には `× 9.80665` で `[N·mm²]` に変換 |
| σ の単位 | spar_calculator は `[kgf/mm²]`、制約値 `sigma_allow_Pa` は `[Pa]` |

### 環境制約

```
scipy がインストール不可（pip 外部ネットワーク遮断）
  → scipy.optimize.brentq の代わりに純 NumPy の二分法を自前実装
  → pymoo も使えないため DE/GA は自前実装

MATLAB R2025b の Global Optimization Toolbox で opt-ATLAS を検証済み
  → Python 移植時は ga() の挙動を意識して実装
```

### 数値安定性

```python
# EI が 0 に近い場合のゼロ除算を防ぐ（opt-ATLAS より）
min_EI = 1e-3 * (1/9.80665) * 1e6  # [kgf·mm²]
EI_dist[EI_dist < min_EI] = min_EI

# 断面係数 Z が 0 に近い場合
Z_dist[Z_dist < 1e-3] = 1e-3
```

---

## 付録: ヒアリングログ（主要Q&A）

| 質問 | 回答要約 |
|------|---------|
| 最終的に何を決定したいか | 翼スパン・機速に対して最適 β と積層構成を出力 |
| ユーザーは誰か | WASA OB。後輩設計支援ツールとして共有予定 |
| opt-ATLASとの関係 | 全周積層専用の先行実装。ロジックは正しい。Hemere は集中積層対応への拡張 |
| 集中積層の設計変数 | カバー角 × ply 数の組み合わせ。2ply毎に5°ずつずらして積層。最大16層 |
| surrogateモデルの状況 | 検証精度は良好だが実計算で変な値が出る。EI_req 逆算の問題の可能性 |
| マンドレル径の扱い | 今は固定パラメータ。将来的にスリーホープ規格径から選択 |
| EI_req 逆算の方法 | 未確立。強度計算と曲率分布の問題。数式化できていない |
| 積層逆算の方法 | 解析的逆算（二分法）で最小 n_cap を求める。製造可否チェックもここで |
| scipy が使えるか | pip 外部遮断のため使用不可。純 NumPy で実装 |

---

---

## 12. Step 4 実装の課題と対策

### バグ修正内容（4ファイル完了）

#### 課題①: spar_calculator の返値不足
```python
# Before:
return total_EIx_kgf, total_weight_kg_m, total_thickness_mm

# After:
total_I_mm4 = float(np.sum(Ix))          # 断面二次モーメント合計
D_outer_mm  = float(outer_dia[-1])       # 最外層外径
return total_EIx_kgf, total_weight_kg_m, total_thickness_mm, total_I_mm4, D_outer_mm
```

#### 課題②: structural_analyzer の scipy 依存
```python
# Before:
from scipy.integrate import cumulative_trapezoid

# After:
@staticmethod
def _cumtrapz(y: np.ndarray, x: np.ndarray, initial: float = 0.0) -> np.ndarray:
    """scipy.integrate.cumulative_trapezoid の純NumPy実装"""
    dx = np.diff(x)
    trapz = (y[:-1] + y[1:]) * dx / 2.0
    return np.concatenate([[initial], np.cumsum(trapz)])
```

#### 課題③: check_strength の E固定
```python
# Before:
E_assumed = 100e9  # [Pa] 固定

# After:
def check_strength(self, moment_Nm: np.ndarray,
                   I_dist_mm4: np.ndarray,
                   D_outer_dist_mm: np.ndarray) -> np.ndarray:
    """σ = M * c / I  （c = 外半径）"""
    I_m4 = I_dist_mm4 * 1e-12
    c_m  = D_outer_dist_mm / 1000.0 / 2.0
    stress = np.abs(moment_Nm) * c_m / (I_m4 + 1e-18)
    return stress / 1e6  # MPa
```

#### 課題④: design_integrator_v3 のキー名不整合
```python
# Before:
s['Actual_EI'], s['Actual_Weight']  # ← 実は存在しないキー

# After:
s['EI'], s['Weight']  # ← snap_optimizer が返す正しいキー
solutions[0]          # ← solve() はリストを返すので [0] を取得
```

### 単位変換の追加（design_integrator_v3）

```python
# EI_req は ei_estimator から [N*mm²] で返される
# snap_optimizer は [kgf*mm²] を期待するので変換必須
target_ei_kgf = target_ei_dist / self.g  # N*mm² → kgf*mm²
solutions = self.snap_opt.solve(target_ei_kgf)

# 逆に actual_ei は [kgf*mm²] なので
# compute_deflection に渡す際は [N*mm²] に変換
final_deflection, _ = self.struct.compute_deflection(final_moment, actual_ei * self.g)
```

### 実行状況

2026-03-26 時点での実行:
- パイプラインは起動可能（ただし処理中断）
- snapshot 群（snap_optimizer による逐段探索）は正常に実行されている
- 最終的な検証には ei_estimator.py の実装が必須

---

*このドキュメントは 2026-03-26 時点の状況を記録したものです。*
*開発の進捗に合わせて適宜更新してください。*
