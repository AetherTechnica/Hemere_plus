import numpy as np

class SparCalculator:
    """
    CFRPパイプの重量や曲げ剛性を積層構成から導出するクラス
    """

    def __init__(self):
        # --- プリプレグヤング率 (単位: kgf/mm^2) ---
        # Index対応:
        # 0: 24t_0deg
        # 1: 24t_45deg
        # 2: 24t_90deg
        # 3: 40t_0deg
        # 4: 40t_45deg
        # 5: 40t_90deg
        self.mat_props_kgf = np.array([13000, 1900, 900, 22000, 1900, 800])
        
        # --- 設計定数 ---
        self.gravity = 9.80665
        
        # 層ごとの有効角度幅 (degree)
        # 90度 = 全周 (180度対称モデルの半周分)
        # Index 0,10: 24t (Protect) -> 90deg
        # Index 1:    40t (Torque)        -> 90deg
        # Index 2:    40t (Base)          -> 90deg
        # Index 3~9:  Caps (Gradation)    -> 50deg ~ 20deg
        self.ply_angles = np.array([90, 90, 90, 50, 45, 40, 35, 30, 25, 20, 90])

        # 線密度係数 (g/mm^3)
        # 24t
        self.density_coef_outer = 0.001496
        # 40t
        self.density_coef_inner = 0.001559

        # 1層あたりの厚み (mm)
        self.t_ply_40t = 0.111
        self.t_ply_24t  = 0.125

    def calculate_spec(self, ply_counts: np.ndarray, diameter_mm: float):
        """
        積層数と直径からスペックを計算する。

        Args:
            ply_counts (np.ndarray): 各層の積層数配列
            diameter_mm (float): マンドレル直径 (mm)

        Returns:
            tuple: (
                EI [kgf*mm^2],
                Weight [kg/m],
                TotalThickness [mm],
                I_mm4 [mm^4],       全層合計の断面二次モーメント
                D_outer_mm [mm]     最外層の外径
            )
        """
        n_layers = 11
        
        # 入力チェック
        if len(ply_counts) != n_layers:
            raise ValueError(f"積層構成配列の要素数は {n_layers}である必要があります。受け取った配列数→ {len(ply_counts)}")

        # 1. 厚みの計算
        thickness = ply_counts * self.t_ply_40t
        # 最内・最外のみ厚みを上書き
        thickness[0] = ply_counts[0] * self.t_ply_24t
        thickness[10] = ply_counts[10] * self.t_ply_24t

        # 2. 内径・外径の積算
        inner_dia = np.zeros(n_layers)
        outer_dia = np.zeros(n_layers)
        
        current_inner = diameter_mm
        for i in range(n_layers):
            inner_dia[i] = current_inner
            outer_dia[i] = current_inner + 2 * thickness[i]
            current_inner = outer_dia[i] # 次の層の内径は今の層の外径

        # 3. 断面二次モーメント (I) の計算
        # 部分積層を考慮した慣性モーメントの式
        Ix_factors = (self.ply_angles * np.pi / 360.0 + np.sin(2 * self.ply_angles * np.pi / 180.0) / 4.0) / 16.0
        Ix = (outer_dia**4 - inner_dia**4) * Ix_factors

        # 4. ヤング率 (E) の割り当て
        E_vec_kgf = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1 # 1-based index logic from original code
            
            if idx == 1 or idx == 11:
                # 最内・最外 -> 24t_90 (Mat Idx 2)
                E_vec_kgf[i] = self.mat_props_kgf[2]
            elif idx == 2:
                # トルク層 -> 40t_45 (Mat Idx 4)
                E_vec_kgf[i] = self.mat_props_kgf[4]
            else:
                # 0度層 -> 40t_0 (Mat Idx 3)
                E_vec_kgf[i] = self.mat_props_kgf[3]

        # 5. 剛性 (EI) 計算
        EIx_kgf = Ix * E_vec_kgf
        total_EIx_kgf = np.sum(EIx_kgf)

        # 6. 重量 (Weight) 計算 [kg/m]
        # 断面積(近似) × 角度比
        area_factor = (np.pi / 4.0) * (outer_dia**2 - inner_dia**2) * (self.ply_angles / 90.0)
        
        weights_layer = np.zeros(n_layers)
        for i in range(n_layers):
            idx = i + 1
            if idx == 1 or idx == 11:
                weights_layer[i] = self.density_coef_outer * area_factor[i]
            else:
                weights_layer[i] = self.density_coef_inner * area_factor[i]
        
        total_weight_kg_m = np.sum(weights_layer)
        total_thickness_mm = np.sum(thickness)

        total_I_mm4 = float(np.sum(Ix))   # 全層の断面二次モーメント合計 [mm^4]
        D_outer_mm  = float(outer_dia[-1]) # 最外層外径 [mm]
        return total_EIx_kgf, total_weight_kg_m, total_thickness_mm, total_I_mm4, D_outer_mm

if __name__ == "__main__":
    # --- 簡易動作テスト ---
    calc = SparCalculator()
    
    # テスト用データ: 全部1プライ
    test_ply = np.array([1,2,1,1,1,1,1,1,1,1,1]) #np.array配列で入力
    test_dia = 100.0 # mm
    
    ei, weight, t, _, _ = calc.calculate_spec(test_ply, test_dia)
    
    print(f"--- SparCalculator Test ---")
    print(f"Diameter: {test_dia} mm")
    print(f"Ply     : {test_ply}")
    print(f"EI      : {ei:.4e} kgf mm^2")
    print(f"Weight  : {weight:.4f} kg/m")
    print(f"Thick   : {t:.3f} mm")
