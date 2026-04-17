import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# --- 1. 读取并准备数据 ---
file_path = 'GAL/data/三轮.xlsx'
raw_df = pd.read_excel(file_path)
columns = ['Initial Library', 'Round1', 'Round2', 'Round3']

df_list = []
stats_list = []

for i, col in enumerate(columns):
    if col in raw_df.columns:
        data = raw_df[col].dropna()
        df_list.append(pd.DataFrame({'Iteration': col, 'Value': data}))
        # 计算均值和标准差
        stats_list.append({'x': i, 'mean': data.mean(), 'std': data.std()})

df_melted = pd.concat(df_list, ignore_index=True)

# --- 2. 颜色方案定义 ---
# 基础浅色背景 (来自你的图片风格)
bg_colors = ['#DBEAF0', '#DBEAF0', '#F7D6D6', '#F7D6D6']

# 自动生成对应的深色 (用于 Errorbar 和 Mean point)
def get_darker_color(color, factor=0.7):
    """将颜色变深：factor越小越深"""
    rgb = mcolors.hex2color(color)
    dark_rgb = [x * factor for x in rgb]
    return mcolors.rgb2hex(dark_rgb)

dark_colors = [get_darker_color(c, factor=0.6) for c in bg_colors]

# --- 3. 绘图执行 ---
sns.set_theme(style="ticks")
fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

# 3a. 小提琴图
sns.violinplot(x='Iteration', y='Value', data=df_melted, 
               palette=bg_colors, inner=None, linewidth=1.0, 
               edgecolor='grey', ax=ax)

# 3b. 散点叠加
sns.stripplot(x='Iteration', y='Value', data=df_melted, 
              size=3, color='black', alpha=0.2, jitter=True, ax=ax)

# 3c. 绘制均值和方差线 (使用对应的深色)
for i, s in enumerate(stats_list):
    d_color = dark_colors[i]
    # 绘制 Errorbar (方差线)
    ax.errorbar(s['x'], s['mean'], yerr=s['std'], 
                fmt='none', ecolor=d_color, elinewidth=2, capsize=5, zorder=5)
    # 绘制均值点 (使用对应深色填充)
    ax.plot(s['x'], s['mean'], 'o', markerfacecolor=d_color, 
            markeredgecolor='white', markeredgewidth=1, markersize=8, zorder=6)

# --- 4. 细节修饰 ---
ax.set_yscale('log')
ax.set_ylabel('Fluorescence Intensity (a.u.)', fontsize=12, fontweight='bold')
ax.set_xlabel('')

# 确保坐标轴连续不断开
sns.despine(trim=False)

# 绘制 WT 参考线
wt_value = 2283588 
ax.axhline(y=wt_value, color='#555555', linestyle='--', linewidth=1.2, alpha=0.6)
ax.text(4.2, wt_value * 1.1, f'WT: {wt_value:,}', va='center', ha='right', 
        fontsize=10, color='#555555', fontweight='bold')

ax.set_title('Evolutionary Trajectory of RBS Strength (Mean ± SD)', fontsize=14, pad=20)

plt.tight_layout()
plt.show()