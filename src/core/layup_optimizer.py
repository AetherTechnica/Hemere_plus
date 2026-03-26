import numpy as np
from src.core.spar_calculator import SparCalculator


class LayupOptimizer:
    """
    EI_req と マンドレル直径から、最小重量の積層構成を貪欲法で逆算するクラス。

    積層テンプレート（11層）:
        index: 0    1    2    3    4    5    6    7    8    9   10
        角度:  90°  90°  90°  50°  45°  40°  35°  30°  25°  20°  90°
        材料:  24t  40t  40t  40t  40t  40t  40t  40t  40t  40t  24t
        固定:   1    2    1    ← 設計変数（集中積層）→             1

    アルゴリズム:
        1. 固定層で EI_base を計算
        2. 各設計変数層（index 2〜9）の EI増加量/重量増加量 比（効率）を計算
        3. 最高効率の層を 1 枚追加 → EI 再計算
        4. EI >= EI_req になるまで繰り返し
        5. 座屈（D/t）制約チェック
    """

    # 固定層（変えない）: index 0, 1, 10
    FIXED_PLY = {0: 1, 1: 2, 10: 1}

    # 設計変数層: index 2〜9 の上限枚数
    # index 2（0° 全周）は主要曲げ担当 → 最大 12 枚
    # index 3〜9（集中積層）は製造上の制約から各最大 2 枚（2ply毎に5°ずつという実製造制約）
    VAR_LAYER_MAX = {2: 12, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}

    # 座屈制約: D_outer / 総厚み ≤ この値
    BUCKLING_LIMIT = 150.0

    # 最大積層数（合計）
    MAX_TOTAL_PLIES = 20

    # EI_req 下限（翼端近くで EI が 0 になるのを防ぐ）[kgf*mm^2]
    MIN_EI_REQ = 1e7

    def __init__(self):
        self.calc = SparCalculator()

    def _base_ply_counts(self):
        """固定層のみの初期積層配列を返す"""
        ply = np.zeros(11, dtype=float)
        for idx, n in self.FIXED_PLY.items():
            ply[idx] = n
        return ply

    def _compute_marginal_efficiency(self, ply_counts, diameter_mm):
        """
        各設計変数層を 1 枚追加したときの EI 増加量 / 重量増加量 [kgf*mm^2 / (kg/m)] を計算。

        Returns:
            dict: {layer_index: efficiency, ...}  (追加不可能な層は除く)
        """
        EI_cur, W_cur, T_cur, _, _ = self.calc.calculate_spec(ply_counts, diameter_mm)
        efficiencies = {}

        for idx, max_n in self.VAR_LAYER_MAX.items():
            if ply_counts[idx] >= max_n:
                continue  # 上限に達している
            # 1 枚追加
            ply_next = ply_counts.copy()
            ply_next[idx] += 1
            EI_next, W_next, T_next, _, _ = self.calc.calculate_spec(ply_next, diameter_mm)

            dEI = EI_next - EI_cur
            dW  = W_next  - W_cur

            if dW < 1e-12:
                continue  # 重量が変わらない（ゼロ厚み層等）

            efficiencies[idx] = dEI / dW

        return efficiencies

    def find_min_weight_layup(self,
                              EI_req_kgfmm2: float,
                              diameter_mm: float) -> dict:
        """
        EI_req を満たす最小重量の積層構成を貪欲法で求める。

        Args:
            EI_req_kgfmm2 : 必要曲げ剛性 [kgf*mm^2]
            diameter_mm   : マンドレル直径 [mm]

        Returns:
            dict: {
                'ply_counts'  : np.ndarray[11],  各層の積層数
                'EI_actual'   : float [kgf*mm^2],
                'weight'      : float [kg/m],
                'thickness'   : float [mm],
                'I_mm4'       : float [mm^4],
                'D_outer_mm'  : float [mm],
                'D_t'         : float,           外径 / 総厚み
                'margin_pct'  : float,           (EI_actual - EI_req) / EI_req * 100 [%]
                'feasible'    : bool,
                'reason'      : str,             失敗時の理由
            }
        """
        # EI_req に下限を適用
        EI_req = max(EI_req_kgfmm2, self.MIN_EI_REQ)

        ply = self._base_ply_counts()

        # 初期 EI チェック（固定層のみで足りる場合）
        EI, W, T, I_mm4, D_outer = self.calc.calculate_spec(ply, diameter_mm)

        if EI >= EI_req:
            return self._make_result(ply, EI, W, T, I_mm4, D_outer,
                                     EI_req, diameter_mm, feasible=True,
                                     reason="固定層のみで剛性を満足")

        # 貪欲法: EI_req 達成 かつ D/t ≤ BUCKLING_LIMIT の両方を満たすまで積層追加
        # ※ 積層を追加すると T が増えて D/t が下がるため、座屈制約は追加を続けるほど改善する
        total_plies = int(np.sum(ply))
        for _ in range(100):  # 最大 100 反復（無限ループ防止）
            # 両条件を確認
            dt_ok = T > 0 and (diameter_mm / T) <= self.BUCKLING_LIMIT
            ei_ok = EI >= EI_req

            if ei_ok and dt_ok:
                break  # 両方満たした

            # 各層の追加効率を計算
            efficiencies = self._compute_marginal_efficiency(ply, diameter_mm)

            if not efficiencies:
                # 追加可能な層がない
                dt_ok2 = T > 0 and (diameter_mm / T) <= self.BUCKLING_LIMIT
                if not dt_ok2:
                    return self._make_result(ply, EI, W, T, I_mm4, D_outer,
                                             EI_req, diameter_mm, feasible=False,
                                             reason="積層上限に達したが座屈制約を満たせない")
                return self._make_result(ply, EI, W, T, I_mm4, D_outer,
                                         EI_req, diameter_mm, feasible=False,
                                         reason="積層上限に達したが EI 不足")

            # EI が足りている場合は、重量増加量が最小（D/t改善に必要な最薄）の層を選ぶ
            # EI が足りない場合は、EI/重量効率が最大の層を選ぶ
            if ei_ok and not dt_ok:
                best_idx = min(efficiencies, key=lambda k: efficiencies[k])  # 最小重量追加
            else:
                best_idx = max(efficiencies, key=lambda k: efficiencies[k])  # 最大 EI 効率

            ply[best_idx] += 1
            total_plies   += 1

            EI, W, T, I_mm4, D_outer = self.calc.calculate_spec(ply, diameter_mm)

            # 総積層数チェック
            if total_plies > self.MAX_TOTAL_PLIES:
                ply[best_idx] -= 1
                EI, W, T, I_mm4, D_outer = self.calc.calculate_spec(ply, diameter_mm)
                dt_ok2 = T > 0 and (diameter_mm / T) <= self.BUCKLING_LIMIT
                feasible = (EI >= EI_req) and dt_ok2
                reason = f"最大積層数 {self.MAX_TOTAL_PLIES} に到達"
                return self._make_result(ply, EI, W, T, I_mm4, D_outer,
                                         EI_req, diameter_mm, feasible=feasible,
                                         reason=reason)

        dt_final = T > 0 and (diameter_mm / T) <= self.BUCKLING_LIMIT
        feasible = (EI >= EI_req) and dt_final
        reason = "収束" if feasible else (
            f"EI不足 margin={((EI-EI_req)/EI_req*100):.1f}%" if EI < EI_req
            else f"座屈制約違反 D/t={diameter_mm/T:.1f}"
        )

        return self._make_result(ply, EI, W, T, I_mm4, D_outer,
                                  EI_req, diameter_mm, feasible=feasible,
                                  reason=reason)

    def _make_result(self, ply, EI, W, T, I_mm4, D_outer,
                     EI_req, diameter_mm, feasible, reason):
        D_t = D_outer / T if T > 0 else float('inf')
        margin = (EI - EI_req) / EI_req * 100.0 if EI_req > 0 else 0.0
        return {
            'ply_counts'  : ply.copy(),
            'EI_actual'   : float(EI),
            'weight'      : float(W),
            'thickness'   : float(T),
            'I_mm4'       : float(I_mm4),
            'D_outer_mm'  : float(D_outer),
            'D_t'         : float(D_t),
            'margin_pct'  : float(margin),
            'feasible'    : feasible,
            'reason'      : reason,
        }

    def solve_span(self,
                   EI_req_kgfmm2_dist: np.ndarray,
                   diameter_mm_dist: np.ndarray) -> list:
        """
        スパン全域の EI_req 分布に対して各位置の積層構成を求める。

        Args:
            EI_req_kgfmm2_dist: EI_req 分布 [kgf*mm^2], shape (M,)
            diameter_mm_dist  : マンドレル直径分布 [mm],   shape (M,)

        Returns:
            list[dict]: 各セグメントの solve 結果
        """
        assert len(EI_req_kgfmm2_dist) == len(diameter_mm_dist), \
            "EI_req とマンドレル径の長さが一致しません"

        results = []
        for EI_req, D in zip(EI_req_kgfmm2_dist, diameter_mm_dist):
            res = self.find_min_weight_layup(EI_req, D)
            results.append(res)
        return results


# ===========================================================================
# 簡易テスト
# ===========================================================================
if __name__ == "__main__":
    opt = LayupOptimizer()

    print("=== LayupOptimizer 簡易テスト ===")
    print()

    # テスト1: 固定層のみで充分な場合
    EI_small = 5e8  # [kgf*mm^2]
    res = opt.find_min_weight_layup(EI_small, 100.0)
    print(f"T1 EI_req={EI_small:.0e}:")
    print(f"   ply={res['ply_counts'].tolist()}")
    print(f"   EI_actual={res['EI_actual']:.3e}, weight={res['weight']:.4f} kg/m")
    print(f"   margin={res['margin_pct']:.1f}%, feasible={res['feasible']}")
    print()

    # テスト2: 積層が段階的に増える場合
    EI_mid = 1.5e10  # [kgf*mm^2]
    res = opt.find_min_weight_layup(EI_mid, 100.0)
    print(f"T2 EI_req={EI_mid:.0e}:")
    print(f"   ply={res['ply_counts'].tolist()}")
    print(f"   EI_actual={res['EI_actual']:.3e}, weight={res['weight']:.4f} kg/m")
    print(f"   margin={res['margin_pct']:.1f}%, feasible={res['feasible']}")
    print(f"   D/t={res['D_t']:.1f}, D_outer={res['D_outer_mm']:.1f} mm")
    print()

    # テスト3: 検証済み積層（HANDOVER.md 記載: EI=1.4363e10 kgf*mm^2）
    EI_ref = 1.4363e10
    res = opt.find_min_weight_layup(EI_ref, 120.0)
    print(f"T3 EI_req={EI_ref:.4e} (HANDOVER.md 検証値):")
    print(f"   ply={res['ply_counts'].tolist()}")
    print(f"   EI_actual={res['EI_actual']:.4e}, weight={res['weight']:.4f} kg/m")
    print(f"   margin={res['margin_pct']:.2f}%, feasible={res['feasible']}")
    print()

    # テスト4: 翼端（EI ほぼ 0 → 下限適用）
    EI_near_zero = 0.0
    res = opt.find_min_weight_layup(EI_near_zero, 45.0)
    print(f"T4 EI_req=0 (翼端, 下限 {LayupOptimizer.MIN_EI_REQ:.0e} が適用される):")
    print(f"   ply={res['ply_counts'].tolist()}")
    print(f"   EI_actual={res['EI_actual']:.3e}, weight={res['weight']:.4f} kg/m")
    print(f"   feasible={res['feasible']}, reason={res['reason']}")
