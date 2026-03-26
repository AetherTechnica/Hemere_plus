"""
spar_calculator.py の物理モデル検証スクリプト
================================================
opt-ATLAS の calculateStiffness と比較し、EI 計算の正確性を確認する。

検証ポイント:
  1. opt-ATLAS の R_mm が「半径」か「直径」か
  2. 全周積層での EI 値が opt-ATLAS と一致するか
  3. Ix_factors（カバー角度補正）の式が正しいか
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from src.core.spar_calculator import SparCalculator

# ===========================================================================
# 1. opt-ATLAS の calculateStiffness を Python で再現
# ===========================================================================

def opt_atlas_calculate_stiffness(ply_number, pp, R_mm):
    """
    opt-ATLAS の calculateStiffness をそのまま Python で再現。
    R_mm は opt-ATLAS の変数名のまま使用（半径か直径かは検証対象）。

    積層テンプレート(opt-ATLAS):
      ply_number = [1, 1, 1, n_0deg, 1]
      pp         = [90, 45, 45, 0, 90]
    """
    material = [0, 13000, 1900, 900, 22000, 1900, 800]  # 1-indexed
    PLY_THICKNESS_MM = 0.111
    N_struct = len(ply_number)

    # 厚み計算
    thickness_mm = np.array(ply_number, dtype=float) * PLY_THICKNESS_MM

    # 内径・外径計算（opt-ATLAS のロジックそのまま）
    inner_mm = np.zeros(N_struct)
    outer_mm = np.zeros(N_struct)
    inner_mm[0] = R_mm
    for i in range(1, N_struct):
        inner_mm[i] = inner_mm[i-1] + 2 * thickness_mm[i-1]
    outer_mm = inner_mm + 2 * thickness_mm

    # 断面二次モーメント
    ix_mm4 = (np.pi / 64) * (outer_mm**4 - inner_mm**4)

    # EI 計算
    E_dict = {0: 22000, 45: 1900, 90: 900}  # kgf/mm^2
    eix_kgfmm2 = np.zeros(N_struct)
    for i in range(N_struct):
        angle = pp[i]
        if angle == 0:
            E = 22000
        elif angle == 45:
            E = 1900
        elif angle == 90:
            E = 900
        else:
            E = 900  # フォールバック
        eix_kgfmm2[i] = ix_mm4[i] * E

    EI_total = np.sum(eix_kgfmm2)

    # 断面積
    area_mm2 = np.sum((np.pi / 4) * (outer_mm**2 - inner_mm**2))

    return EI_total, area_mm2, np.sum(ix_mm4), outer_mm[-1], inner_mm, outer_mm, ix_mm4


# ===========================================================================
# 2. Hemere の spar_calculator で全周積層を再現
# ===========================================================================

def hemere_full_wrap(n_0deg, diameter_mm):
    """
    Hemere の spar_calculator を使って「全周積層」を模擬する。
    全周積層 = ply_angles がすべて 90° のとき。

    ただし Hemere の現在の積層テンプレートでは、
    index 3〜9 が集中積層（50°〜20°）になっている。
    ここではカスタム計算で全周積層（全て90°）を再現する。
    """
    calc = SparCalculator()

    # opt-ATLAS と対応させる:
    # opt-ATLAS: pp = [90, 45, 45, 0, 90], ply = [1, 1, 1, n_0deg, 1]
    # Hemere 11層テンプレートの対応:
    #   index 0:  24t_90  → 1枚（保護層）
    #   index 1:  40t_45  → 4枚（トルク層、opt-ATLASの index1+2 に相当）
    #   index 2:  40t_0   → 全周（n_0deg 枚）
    #   index 3〜9: 集中積層（今回は 0 枚にして無効化）
    #   index 10: 24t_90  → 1枚（保護層）
    ply_counts = np.zeros(11)
    ply_counts[0]  = 1        # 保護層内側
    ply_counts[1]  = 4        # トルク層（45°全周 × 4枚）← opt-ATLAS の pp[1]=45×1, pp[2]=45×1
    ply_counts[2]  = n_0deg   # 0度層（全周 = index2 は90°カバー）
    ply_counts[10] = 1        # 保護層外側
    # index 3〜9: 0枚（集中積層なし）

    return calc.calculate_spec(ply_counts, diameter_mm)


# ===========================================================================
# 3. Ix_factor の数式検証
# ===========================================================================

def verify_ix_factor():
    """
    カバー角度 90°（全周）のとき、Ix_factor = π/64 になることを確認。
    これが opt-ATLAS の (π/64) と一致するかチェック。
    """
    angle_deg = 90.0
    angle_rad = angle_deg * np.pi / 180.0

    Ix_factor_hemere = (angle_deg * np.pi / 360.0 + np.sin(2 * angle_rad) / 4.0) / 16.0
    Ix_factor_optATLAS = np.pi / 64.0  # 全周積層の場合

    print("=== Ix_factor 検証（カバー角90°の場合） ===")
    print(f"  Hemere  Ix_factor: {Ix_factor_hemere:.8f}")
    print(f"  opt-ATLAS (π/64): {Ix_factor_optATLAS:.8f}")
    print(f"  一致: {np.isclose(Ix_factor_hemere, Ix_factor_optATLAS)}")
    print()

    # 他のカバー角度でも確認
    print("  カバー角度別 Ix_factor:")
    for angle in [20, 30, 35, 40, 45, 50, 90]:
        angle_r = angle * np.pi / 180.0
        f = (angle * np.pi / 360.0 + np.sin(2 * angle_r) / 4.0) / 16.0
        ratio = f / Ix_factor_optATLAS  # 全周比
        print(f"    {angle:3d}° → Ix_factor = {f:.6f}  (全周比 {ratio:.4f})")
    print()


# ===========================================================================
# 4. opt-ATLAS の R_mm が半径か直径かの検証
# ===========================================================================

def verify_R_mm_interpretation():
    """
    opt-ATLAS の R_mm = 90 が「半径」か「直径」かを確かめる。

    方法:
      opt-ATLAS の calculateStiffness を R_mm=90 で呼ぶ。
      - R_mm=90 を「直径」として扱う場合: 内径90mm
      - R_mm=90 を「半径」として扱う場合: 内径180mm

    Hemere の spar_calculator では diameter_mm（直径）を入力とする。
    したがって:
      - R_mm が直径なら: diameter_mm = R_mm = 90
      - R_mm が半径なら: diameter_mm = 2 × R_mm = 180
    """
    # opt-ATLAS 条件3（高速ショートスパン）の翼根セグメント
    # n_0deg = 6（条件3の最適解より）
    n_0deg = 6
    ply_number_atlas = [1, 1, 1, n_0deg, 1]
    pp_atlas = [90, 45, 45, 0, 90]
    R_mm_atlas = 90.0  # opt-ATLAS の mandrel_R_dist_mm の翼根値

    EI_atlas, _, _, D_outer_atlas, inner_mm, outer_mm, _ = \
        opt_atlas_calculate_stiffness(ply_number_atlas, pp_atlas, R_mm_atlas)

    print("=== opt-ATLAS の R_mm 解釈検証 ===")
    print(f"  R_mm = {R_mm_atlas} [mm]（opt-ATLASのまま）")
    print(f"  計算後の内径（inner_mm[0]）: {inner_mm[0]:.3f} mm")
    print(f"  計算後の外径（outer_mm[-1]）: {outer_mm[-1]:.3f} mm")
    print(f"  EI(opt-ATLAS): {EI_atlas:.4e} kgf*mm^2")
    print()

    # Hemere で同じ条件を「直径=90mm」で計算
    print("  --- Hemere で diameter_mm=90（R_mm=90 を直径として解釈） ---")
    EI_hem_90, w_90, t_90 = hemere_full_wrap(n_0deg, 90.0)
    print(f"  EI: {EI_hem_90:.4e} kgf*mm^2")
    print(f"  Weight: {w_90:.4f} kg/m")

    # Hemere で同じ条件を「直径=180mm」で計算（R=90を半径として解釈）
    print()
    print("  --- Hemere で diameter_mm=180（R_mm=90 を半径として解釈） ---")
    EI_hem_180, w_180, t_180 = hemere_full_wrap(n_0deg, 180.0)
    print(f"  EI: {EI_hem_180:.4e} kgf*mm^2")
    print(f"  Weight: {w_180:.4f} kg/m")

    print()
    print("  -> opt-ATLAS の EI に近い方が正しい解釈")
    diff_90  = abs(EI_atlas - EI_hem_90)  / EI_atlas * 100
    diff_180 = abs(EI_atlas - EI_hem_180) / EI_atlas * 100
    print(f"  diameter=90  との差: {diff_90:.2f}%")
    print(f"  diameter=180 との差: {diff_180:.2f}%")
    print()


# ===========================================================================
# 5. 引き継ぎ資料の検証値の再確認
# ===========================================================================

def verify_handover_value():
    """
    引き継ぎ資料に記載の検証値:
      D=120mm（内径）, ply=[1,4,6,1,0,0,0,0,1,1,1]
      → EI = 1.4363×10¹⁰ kgf*mm^2（エクセルと一致）
    """
    calc = SparCalculator()
    ply_counts = np.array([1, 4, 6, 1, 0, 0, 0, 0, 1, 1, 1], dtype=float)
    diameter_mm = 120.0

    EI, weight, thickness = calc.calculate_spec(ply_counts, diameter_mm)

    print("=== 引き継ぎ資料の検証値確認 ===")
    print(f"  入力: diameter={diameter_mm}mm, ply={ply_counts.astype(int).tolist()}")
    print(f"  EI:     {EI:.4e} kgf*mm^2")
    print(f"  期待値: 1.4363e+10 kgf*mm^2")
    print(f"  誤差:   {abs(EI - 1.4363e10) / 1.4363e10 * 100:.3f}%")
    print(f"  Weight: {weight:.4f} kg/m")
    print(f"  Thick:  {thickness:.3f} mm")

    # 単位変換
    EI_Nmm2 = EI * 9.80665
    print(f"  EI [N*mm^2]: {EI_Nmm2:.4e}")
    print()


# ===========================================================================
# 6. opt-ATLAS 全体フローの確認値との比較
# ===========================================================================

def verify_atlas_condition3():
    """
    opt-ATLAS 条件3の最適解（全セグメント）で EI を比較。
    条件3: スパン22m（全スパン）, 速度11m/s, 翼根R=90mm, 翼端R=40mm
    最適積層（8分割）: [6, 8, 7, 5, 4, 3, 1, 1]
    """
    ply_number_atlas_base = [1, 1, 1, 0, 1]
    pp_atlas = [90, 45, 45, 0, 90]
    R_dist = np.linspace(90, 40, 8)  # opt-ATLAS の R_mm（翼根→翼端）
    n_0deg_opt = [6, 8, 7, 5, 4, 3, 1, 1]  # 最適解

    print("=== opt-ATLAS 条件3 セグメント別 EI 比較 ===")
    print(f"  {'Seg':>3} {'R_mm':>8} {'n_0deg':>8} {'EI_atlas[kgf*mm^2]':>22} {'EI_hemere[kgf*mm^2]':>22} {'誤差%':>8}")
    print("  " + "-" * 80)

    for j in range(8):
        n = n_0deg_opt[j]
        R = R_dist[j]
        ply_num = ply_number_atlas_base.copy()
        ply_num[3] = n

        EI_atlas, _, _, _, _, _, _ = opt_atlas_calculate_stiffness(ply_num, pp_atlas, R)

        # Hemere で diameter=R として比較（直径解釈）
        EI_hem, _, _ = hemere_full_wrap(n, R)

        diff = abs(EI_atlas - EI_hem) / EI_atlas * 100 if EI_atlas > 0 else float('inf')
        print(f"  {j:>3} {R:>8.1f} {n:>8} {EI_atlas:>22.4e} {EI_hem:>22.4e} {diff:>8.3f}%")

    print()


# ===========================================================================
# メイン実行
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  spar_calculator.py 物理モデル検証")
    print("=" * 70)
    print()

    verify_ix_factor()
    verify_R_mm_interpretation()
    verify_handover_value()
    verify_atlas_condition3()

    print("検証完了。上記の結果をもとに問題箇所を特定してください。")
