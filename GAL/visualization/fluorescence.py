import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import chardet

# ==========================================
# 1. 加载与处理数据
# ==========================================
file_path = 'GAL/data/两轮0321.csv'

with open(file_path, 'rb') as f:
    result = chardet.detect(f.read(10000))
    encoding = result['encoding']

df = pd.read_csv(file_path, encoding=encoding)
# 提取第二列：荧光值
raw_values = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values

threshold = 500000
low_data_all = raw_values[raw_values < threshold]
high_data = raw_values[raw_values >= threshold]

# --- 关键修改：左图只取 120 个样本 ---
if len(low_data_all) > 120:
    low_data = np.random.choice(low_data_all, 120, replace=False)
else:
    low_data = low_data_all

# ==========================================
# 2. 绘图配置 (Premium Equal Ratio Style)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 关键：移除 width_ratios，让两图大小 1:1 绝对一致
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6)) 

# 配色方案
color_low = '#4575b4'   
color_high = '#d73027'  

# --- 绘制左图：Low fluorescence value data ---
# 随机生成 X 轴坐标 (0-120)
x_low_rand = np.random.uniform(2, 118, len(low_data)) 

ax0.scatter(x_low_rand, low_data, s=50, color=color_low, 
            edgecolor='white', linewidth=0.6, alpha=0.65)

# 标题与坐标轴 (严格沿用)
ax0.set_title('Low fluorescence value data', fontsize=16, fontweight='bold', pad=25)
ax0.set_ylabel('Fluorescence Intensity (a.u.)', fontsize=13, fontweight='bold')

# 横坐标轴精准设置：0-120
ax0.set_xlim(0, 120)
ax0.set_xticks(np.arange(0, 121, 20)) 
ax0.set_xlabel('Sample Index (0-120)', fontsize=12, labelpad=12)


# --- 绘制右图：Medium and high fluorescence value data ---
# 随机生成 X 轴坐标 (0-20)
x_high_rand = np.random.uniform(1, 19, len(high_data))

ax1.scatter(x_high_rand, high_data, s=80, color=color_high, 
            edgecolor='white', linewidth=0.8, alpha=0.8)

# 标题 (严格沿用)
ax1.set_title('Medium and high fluorescence value data', fontsize=16, fontweight='bold', pad=25)

# 横坐标轴精准设置：0-20
ax1.set_xlim(0, 20)
ax1.set_xticks(np.arange(0, 21, 5)) 
ax1.set_xlabel('Sample Index (0-20)', fontsize=12, labelpad=12)


# --- 细节美化 ---
for ax in [ax0, ax1]:
    # 移除上方和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 水平网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#bdc3c7')
    ax.set_axisbelow(True)
    
    # 强制不使用科学计数法
    ax.ticklabel_format(style='plain', axis='y')
    ax.tick_params(axis='both', which='major', labelsize=11)

# 在右图添加阈值参考线
ax1.axhline(y=threshold, color=color_high, linestyle=':', alpha=0.3, linewidth=1.5)

plt.tight_layout(pad=4.0)

# ==========================================
# 3. 导出高质量文件
# ==========================================
plt.savefig('GAL_EqualRatio_120Samples.svg', format='svg', bbox_inches='tight')
plt.savefig('GAL_EqualRatio_120Samples.png', dpi=400, bbox_inches='tight')

plt.show()

print(f"绘图完成！左右两图大小已保持 1:1 一致。")