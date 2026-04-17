import pandas as pd
import numpy as np
import os

# ================= 1. One-hot 编码核心函数 =================
def extract_one_hot_features(spacer, max_len=8):
    """
    将 6-8nt 的 spacer 转换为 One-hot 编码。
    采用右对齐（靠近 AUG 的位置对齐），空位用全 0 向量填充 (Padding)。
    """
    # 碱基映射字典
    base_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'T': 1} # T 视为 U
    
    spacer = str(spacer).upper().strip()
    s_len = len(spacer)
    
    # 初始化全 0 矩阵 (max_len x 4)
    # 序列变为：[Pad, Pad, Base, Base, Base, Base, Base, Base] (如果是6nt)
    one_hot = np.zeros((max_len, 4))
    
    # 填充逻辑：从后往前填，保证 spacer 的最后一位始终在 one_hot 的最后一行
    # 这样对于模型来说，Position -1 永远是 AUG 之前的那个碱基
    for i in range(s_len):
        base = spacer[s_len - 1 - i]
        if base in base_map:
            # 计算在 one_hot 矩阵中的行索引
            row_idx = max_len - 1 - i
            one_hot[row_idx, base_map[base]] = 1.0
            
    # 将矩阵拉平为一维向量 (8 * 4 = 32 维)
    flatten_features = one_hot.flatten()
    
    # 生成特征列名，例如 P1_A, P1_U, ..., P8_C
    # 注意这里的 P8 是最靠近 AUG 的位置
    feature_names = []
    for pos in range(1, max_len + 1):
        for b in ['A', 'U', 'G', 'C']:
            feature_names.append(f"Pos{pos}_{b}")
            
    return pd.Series(flatten_features, index=feature_names)

# ================= 2. 主程序 =================
def main():
    input_file = "spacer_data.csv"
    output_file = "spacer_onehot_features.csv"
    
    if not os.path.exists(input_file):
        print(f"未找到文件 {input_file}，请确保文件中包含 'spacer' 列。")
        return

    df = pd.read_csv(input_file)
    
    print(f"检测到 {len(df)} 条序列，正在进行 One-hot 编码 (长度 6-8nt)...")
    
    # 1. 提取 One-hot 特征
    # 我们设定 max_len=8，涵盖所有 6, 7, 8bp 的情况
    one_hot_df = df['spacer'].apply(lambda x: extract_one_hot_features(x, max_len=8))
    
    # 2. 拼接原始数据（如荧光值）与新特征
    final_df = pd.concat([df, one_hot_df], axis=1)
    
    # 3. 保存
    final_df.to_csv(output_file, index=False)
    print(f"处理完成！One-hot 特征已保存至 {output_file}")
    
    # --- 简易可视化：查看位置权重偏好 ---
    if 'fluorescence' in df.columns:
        print("\n正在分析各位置碱基与表达量的相关性...")
        # 计算所有 One-hot 列与荧光值的相关系数
        oh_cols = one_hot_df.columns
        corrs = []
        for col in oh_cols:
            corrs.append(final_df[col].corr(final_df['fluorescence']))
        
        # 绘图
        plt.figure(figsize=(12, 5))
        plt.bar(oh_cols, corrs, color='steelblue')
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel("Pearson Correlation with Fluorescence")
        plt.title("Sequence Position Preference (One-hot Analysis)")
        plt.tight_layout()
        plt.show()

import matplotlib.pyplot as plt
if __name__ == "__main__":
    main()