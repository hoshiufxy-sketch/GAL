import pandas as pd
import numpy as np
import RNA
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 环境配置 =================
UPSTREAM_SEQ = "TTTGTTTAACTTTAAAGGAG" 
DOWNSTREAM_SEQ = "atgGTTAGCAAAGGTGAAGAACTGTTTAC"
SD_CORE = "AAGGAG" 
ANTI_SD = "ACCUCC" 

MW_MAP = {'A': 135.13, 'G': 151.13, 'C': 111.10, 'U': 112.09}
STACK_MAP = {'A': -7.6, 'G': -8.2, 'C': -6.8, 'U': -6.2}

# ================= 2. 健壮的特征提取 =================
def extract_complete_features(spacer):
    try:
        if pd.isna(spacer) or str(spacer).strip() == "":
            return pd.Series([0]*12)
        
        spacer = str(spacer).strip().upper().replace("T", "U")
        s_len = len(spacer)
        full_rna = (UPSTREAM_SEQ + spacer + DOWNSTREAM_SEQ).replace("T", "U")
        
        # A. 物理特征
        spacing = s_len
        spacing_penalty = (s_len - 7) ** 2 
        total_mw = sum(MW_MAP.get(b, 0) for b in spacer)
        purine_ratio = (spacer.count('A') + spacer.count('G')) / s_len if s_len > 0 else 0
        total_stacking = sum(STACK_MAP.get(b, 0) for b in spacer)
        gc_spacer = (spacer.count('G') + spacer.count('C')) / s_len if s_len > 0 else 0
        
        # B. 局部异质性
        mid = s_len // 2
        gc_front = (spacer[:mid].count('G') + spacer[:mid].count('C')) / mid if mid > 0 else 0
        gc_back = (spacer[mid:].count('G') + spacer[mid:].count('C')) / (s_len - mid) if (s_len - mid) > 0 else 0
        
        # C. 结构特征
        (ss, mfe) = RNA.fold(full_rna)
        duplex = RNA.duplexfold(SD_CORE.replace("T","U"), ANTI_SD)
        dg_rbs = duplex.energy
        
        sd_start = full_rna.find(SD_CORE.replace("T","U"))
        atg_start = full_rna.find("AUG")
        
        if sd_start != -1 and atg_start != -1:
            rbs_struct = ss[sd_start : sd_start + len(SD_CORE)]
            atg_struct = ss[atg_start : atg_start + 3]
            rbs_acc = rbs_struct.count('.') / len(SD_CORE)
            atg_acc = atg_struct.count('.') / 3
        else:
            rbs_acc, atg_acc = 0, 0

        return pd.Series({
            'Spacing': spacing, 'Spacing_Penalty': spacing_penalty,
            'Total_MW': total_mw, 'Purine_Ratio': purine_ratio,
            'Total_Stacking': total_stacking, 'GC_Front': gc_front,
            'GC_Back': gc_back, 'DG_total': mfe, 'DG_RBS': dg_rbs,
            'RBS_acc': rbs_acc, 'ATG_acc': atg_acc, 'GC_spacer': gc_spacer
        })
    except:
        return pd.Series([0]*12)

# ================= 3. 数据处理与类型强制转换 =================
def run_full_analysis(input_csv, output_csv):
    # 读取数据，显式指定分隔符（如果是 tab 键分割请用 \t）
    df = pd.read_csv(input_csv)
    
    # --- 关键修复：强制转换荧光值为数字 ---
    # errors='coerce' 会把无法转换的脏数据变成 NaN
    df['fluorescence'] = pd.to_numeric(df['fluorescence'], errors='coerce')
    df = df.dropna(subset=['fluorescence']) # 删掉荧光值缺失的行
    
    print(f"数据读取成功，样本数: {len(df)}")
    
    # 提取特征
    print("正在提取特征...")
    feature_df = df['spacer'].apply(extract_complete_features)
    
    # 合并
    combined_df = pd.concat([df, feature_df], axis=1)
    combined_df.to_csv(output_csv, index=False)
    
    # ================= 4. 绘图逻辑优化 =================
    # 包含荧光值在内的所有数值列
    cols_to_corr = ['fluorescence', 'Spacing', 'Spacing_Penalty', 'Total_MW', 
                    'Purine_Ratio', 'Total_Stacking', 'GC_Front', 'GC_Back', 
                    'DG_total', 'DG_RBS', 'RBS_acc', 'ATG_acc', 'GC_spacer']
    
    corr_matrix = combined_df[cols_to_corr].corr()

    plt.figure(figsize=(12, 10))
    
    # 移除了 mask，现在你会看到完整的正方形矩阵
    sns.heatmap(corr_matrix, 
                annot=True,       # 显示数字
                cmap='RdBu_r',    # 颜色
                center=0,         # 0为中心
                fmt=".2f",        # 保留两位小数
                square=True,      # 每个格子正方形
                linewidths=.5)    # 格子间距
    
    plt.title("Correlation Analysis: Features vs Fluorescence")
    plt.tight_layout()
    plt.show()

# ================= 5. 启动 =================
if __name__ == "__main__":
    # 替换为你实际的文件名
     run_full_analysis('spacer_data.csv', 'spacer_ViennaRNA_features.csv')
 