import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据准备 (基于之前的计算结果) ---
labels = ['Random', ' AL', 'GAL (Round2)', 'GAL (Round3)']
e_al_avg = (27641 + 614593 + 908070) / 3 # 基于 100%, 60%, 40% 失败率推导
values = [27641, e_al_avg, 1495022, 2598577]
wt_value = 2283588

# --- 2. 绘图设置 ---
plt.style.use('seaborn-v0_8-muted') # 使用清爽的默认配色
fig, ax = plt.subplots(figsize=(8, 6), dpi=110) # 适中的图片大小

# 绘制柱状图，通过 width 参数让柱子变窄
bars = ax.bar(labels, values, color=['#AAB7B8', '#7FB3D5', '#EC7063', "#F4566B"], 
              edgecolor='white', linewidth=1, width=0.5)

# --- 3. 坐标轴优化：对数坐标 ---


# --- 4. 标注数值 ---
# 对数轴下的标注位置需要微调，乘法偏移比加法偏移更自然
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

# --- 5. 装饰与参考线 ---
ax.axhline(y=wt_value, color='#2E86C1', linestyle='--', linewidth=1.5, alpha=0.8, 
           label=f'Wild Type (WT): {wt_value:,}')

ax.set_ylabel('Expected Expression Value (Log Scale)', fontsize=12)


# 设置纵轴范围，给顶部的数字标注留出空间


# 隐藏多余边框，让图表更清爽
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper left', frameon=True, fontsize=10)
ax.grid(axis='y', linestyle=':', alpha=0.5) # 仅保留横向网格

plt.tight_layout()
plt.show()