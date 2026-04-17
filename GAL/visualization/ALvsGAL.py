import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 载入数据
# 确保文件名为 'ALvsGAL.xlsx'，列名包含 Round, Method, Model, Feature_Ty, R2
df = pd.read_excel('GAL spacer/data/ALvsGAL.xlsx')

# 数据预处理：统一模型名称并创建绘图标签
df['Model'] = df['Model'].replace({'xgboost': 'XGBoost', 'gp': 'GP', 'dnn': 'DNN'})
df['Series'] = df['Model'] + " (" + df['Method'] + ")"

# 2. 设置学术绘图风格
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 3. 创建 1行3列 布局，共享纵轴
fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=True, dpi=300)
feature_types = ['onehot', 'ViennaRNA', 'NT-v2']

# 4. 定义调色板：GAL系列用深色/实线，AL系列用同色系的浅色/虚线
palette = {
    'XGBoost (GAL)': '#D62728', 'XGBoost (AL)': '#FF9896', # 红色系
    'GP (GAL)': '#2CA02C',      'GP (AL)': '#98DF8A',      # 绿色系
    'DNN (GAL)': '#1F77B4',     'DNN (AL)': '#AEC7E8'       # 蓝色系
}

# 5. 循环绘制每个特征集的子图
for i, f_type in enumerate(feature_types):
    ax = axes[i]
    subset = df[df['Feature_Type'] == f_type]
    
    # 绘制折线
    sns.lineplot(
        data=subset, x="Round", y="R2", hue="Series", 
        style="Method", markers=True, dashes=False, 
        palette=palette, ax=ax, linewidth=2.5, markersize=7.5 # 调大了点的大小
    )
    
    # --- 核心修改 1：移除横坐标标题 ---
    ax.set_xlabel("", fontsize=0) # 或者使用 ax.xaxis.label.set_visible(False)
    
    ax.set_ylim(-0.35, 1.05) 
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    #ax.axhline(y=0.9, color='#7F7F7F', linestyle='--', alpha=0.5, linewidth=1.2)
    
    ax.set_title(f" {f_type}", fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Initial', 'Round 1', 'Round 2', 'Round 3'])
    
    if i == 0:
        ax.set_ylabel("$R^2$", fontsize=13)
    
    ax.get_legend().remove()
# 获取原始句柄和标签
raw_handles, raw_labels = axes[0].get_legend_handles_labels()

# 过滤逻辑：只保留我们在 palette 中定义的 Series 名称，去掉 "Series" 和 "Method" 标题
filtered_handles = []
filtered_labels = []
for h, l in zip(raw_handles, raw_labels):
    if l in palette.keys():
        filtered_handles.append(h)
        filtered_labels.append(l)

# 放置全局图例
# ncol=3 表示排成两行（共6个模型项），这样不容易超出左右边界
fig.legend(filtered_handles, filtered_labels, 
           loc='lower center', 
           ncol=3, 
           bbox_to_anchor=(0.5, 0), # 调整 y 值，使其刚好在画布底部
           frameon=True, 
           fontsize=11)

plt.suptitle("Multi-Model Benchmarking: GAL Framework vs. Vanilla Active Learning", 
             fontsize=18, fontweight='bold', y=0.98)

# --- 修改项 3: 预留底部空间 ---
# rect=[左, 下, 右, 上] 这里的 0.12 为底部预留了 12% 的高度给图例
plt.tight_layout(rect=[0, 0.12, 1, 0.95])

plt.savefig("Full_Model_Comparison_Final.png", bbox_inches='tight')
plt.show()