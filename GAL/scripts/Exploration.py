import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
import shap
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
from sklearn.feature_selection import RFE

import RNA
# ================= 1. 参数与生物物理常量配置 =================
UPSTREAM_SEQ = "TTTGTTTAACTTTAAGAAGGAGA" 
DOWNSTREAM_SEQ = "atgGTTAGCAAAGGTGAAGAACTGTTTAC"
SD_CORE = "AAGAAGGAGA"
ANTI_SD = "ACCUCCUUA"

# 物理化学参数映射
# 分子量 (Reflecting steric bulk)
MW_MAP = {'A': 135.13, 'G': 151.13, 'C': 111.10, 'U': 112.09}
# 碱基堆叠能 (kcal/mol, Reflecting local rigidity/flexibility)
STACK_MAP = {'A': -7.6, 'G': -8.2, 'C': -6.8, 'U': -6.2}
# ================= 2. 核心特征提取函数 =================
def extract_complete_features(spacer):
    try:
        if pd.isna(spacer): return pd.Series([np.nan]*12)
   
        spacer = str(spacer).strip().upper().replace("T", "U")
        s_len = len(spacer)
        full_rna = (UPSTREAM_SEQ + spacer + DOWNSTREAM_SEQ).replace("T", "U")
         
        # --- A. 空间与物理特征 (Physical/Steric Features) ---
        spacing = s_len
        spacing_penalty = (s_len - 7) ** 2  # 偏离最优间距7nt的惩罚
        total_mw = sum(MW_MAP.get(b, 0) for b in spacer)
        purine_count = spacer.count('A') + spacer.count('G')
        purine_ratio = purine_count / s_len  # 嘌呤(双环)占比越高，空间越拥挤
        
        # --- B. 局部刚性特征 (Rigidity/Stacking) ---
        total_stacking = sum(STACK_MAP.get(b, 0) for b in spacer)
        
        # --- C. 序列异质性特征 (Position-Specific GC) ---
        mid = s_len // 2
        gc_front = (spacer[:mid].count('G') + spacer[:mid].count('C')) / mid
        gc_back = (spacer[mid:].count('G') + spacer[mid:].count('C')) / (s_len - mid)
        
        # --- D. 热力学特征 (Thermodynamic Features) ---
        # 1. DG_total (MFE)
        (ss, mfe) = RNA.fold(full_rna)
        
        # 2. DG_RBS (结合能)
        duplex = RNA.duplexfold(SD_CORE.replace("T","U"), ANTI_SD)
        dg_rbs = duplex.energy
        
        # 3. 可及性 (Accessibility - 基于 MFE 结构)
        rbs_start = full_rna.find(SD_CORE.replace("T","U"))
        atg_start = full_rna.find("AUG")
        
        rbs_struct = ss[rbs_start : rbs_start + len(SD_CORE)]
        atg_struct = ss[atg_start : atg_start + 3]
        
        rbs_acc = rbs_struct.count('.') / len(SD_CORE)
        atg_acc = atg_struct.count('.') / 3
        
        return pd.Series([
            spacing, spacing_penalty, total_mw, purine_ratio, 
            total_stacking, gc_front, gc_back, 
            mfe, dg_rbs, rbs_acc, atg_acc, (spacer.count('G') + spacer.count('C')) / s_len
        ])

    except Exception as e:
        print(f"Error processing {spacer}: {e}")
        return pd.Series([np.nan]*12)

# 读取 Excel 文件
df = pd.read_csv('GAL/data/spacer_advanced_features2.csv', header=0,)## nrows=99
print(df.head())  # 查看前几行
print(df.columns)  # 检查列名
# 提取输入特征（Z列到BH列）
X = df.iloc[:, 2:] # 25列，BH列是第60列 去掉values!!!16380

# 提取输出值（C列）
y = df.iloc[:, 1] # C列是第2列
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 初始化 XGBoost 回归模型
model = xgb.XGBRegressor(n_estimators=300,learning_rate=0.05, gamma=0.8, max_depth=10,random_state=42,)

# 训练模型
model.fit(X_train, y_train)

# 执行 RFE 特征选择（选前n个特征，可根据需要修改）
selector = RFE(model, n_features_to_select=30, step=1)
selector = selector.fit(X, y)

# 取被选中的特征名称
selected_feature_names = X.columns[selector.get_support()]

# 重新拟合模型以获取这些特征的 importance（用筛选后的特征）
X_selected = X[selected_feature_names]
# ================= 5. Ensemble 训练 (核心改动) =================
n_ensemble = 5
ensemble_models = []
print(f"正在训练 {n_ensemble} 个 Bootstrap 集成模型...")

for i in range(n_ensemble):
    # 有放回抽样制造差异
    idx = np.random.choice(len(X_selected), size=len(X_selected), replace=True)
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, gamma=0.8, max_depth=10, random_state=i)
    model.fit(X_selected.iloc[idx], y.iloc[idx])
    ensemble_models.append(model)

# ================= 6. Ensemble 预测与评估 =================
def predict_ensemble(models, data):
    # 获取所有子模型的预测值
    all_preds = np.array([model.predict(data) for model in models])
    # 返回均值 mu 和 标准差 sigma
    return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)
import itertools
import itertools
# ================= 7. 定义采集函数 (Acquisition Functions) =================
from scipy.stats import norm

def get_acquisition_scores(mu, sigma, y_max, strategy='UCB', kappa=30):
    """
    计算主动学习采集分数
    mu: 预测均值
    sigma: 预测标准差 (不确定性)
    y_max: 当前已知数据的最大值
    """
    if strategy == 'Greedy':
        # 纯开发：只看预测均值
        return mu
    
    elif strategy == 'UCB':
        # 置信上限：兼顾高产(mu)与探索(sigma)
        return mu + kappa * sigma
    
    elif strategy == 'EI':
        # 期望改善：计算超越 y_max 的期望值
        # z 是标准化后的改善程度
        improvement = mu - y_max
        z = improvement / (sigma + 1e-9)
        return improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    return mu
# ================= 8. 生成 6-8bp 全量虚拟序列池 =================
print("正在生成 6-8bp 全量虚拟序列空间 (共约 8.6 万条)...")
bases = ['A', 'T', 'G', 'C']
all_candidate_spacers = []

# 1. 穷举所有 6, 7, 8 bp 的组合
for length in [6, 7, 8]:
    print(f"正在生成 {length}bp 组合...")
    # 注意：如果你的特征提取函数内部会将 T 换成 U，这里用 T 或 U 都可以
    perms = [''.join(p) for p in itertools.product(bases, repeat=length)]
    all_candidate_spacers.extend(perms)

total_count = len(all_candidate_spacers)
print(f"生成完毕，总计候选序列数: {total_count}")

# 2. 批量提取特征 (8.6万条序列大约需要 5-10 分钟，具体取决于 CPU)
print("正在为全量虚拟池提取特征，请稍候...")
virtual_data = []

for i, seq in enumerate(all_candidate_spacers):
    features = extract_complete_features(seq)
    virtual_data.append(features)
    
    # 每处理 5000 条打印一次进度，防止程序看起来像卡死了
    if (i + 1) % 5000 == 0:
        print(f"进度: {i + 1} / {total_count} ({(i + 1)/total_count*100:.1f}%)")

# 转换为 DataFrame
X_virtual = pd.DataFrame(virtual_data)
# 必须确保列名与训练时的 X 完全一致
X_virtual.columns = X.columns 

# 3. 筛选特征：只取 RFE 选出的特征
X_virtual_selected = X_virtual[selected_feature_names]

# 4. 集成模型预测
print("正在使用集成模型进行全景观预测...")
mu, sigma = predict_ensemble(ensemble_models, X_virtual_selected)

# ================= 7. 推荐策略 (Greedy, UCB, EI) - 修复版 =================
y_max = y.max()  
strategies = ['Greedy', 'UCB', 'EI']
final_recommendations = {}

print("\n" + "="*50)
print("🚀 基于 6-8bp 全局景观扫描的序列推荐 (Top 5)")
print("="*50)

# 此时的 candidate_spacers 对应的是全部 8.6 万条序列
candidate_spacers = all_candidate_spacers 

for strat in strategies:
    # 计算得分
    scores = get_acquisition_scores(mu, sigma, y_max, strategy=strat)
    
    # 找到全量空间中分数最高的前 5 个相对位置
    best_relative_indices = np.argsort(scores)[-5:][::-1]
    
    rec_results = []
    for rel_idx in best_relative_indices:
        # 获取该位置对应的原始特征，方便观察为什么它被选中
        row_features = X_virtual.iloc[rel_idx]
        
        rec_results.append({
            'spacer': candidate_spacers[rel_idx],
            'length': len(candidate_spacers[rel_idx]),
            'Predicted_Mean': mu[rel_idx],
            'Uncertainty_Sigma': sigma[rel_idx],
            'atg_acc': row_features['atg_acc'] if 'atg_acc' in row_features else "N/A",
            'mfe': row_features['mfe'] if 'mfe' in row_features else "N/A",
            'Strategy_Score': scores[rel_idx]
        })
    
    rec_df = pd.DataFrame(rec_results)
    final_recommendations[strat] = rec_df
    
    print(f"\n策略【{strat}】推荐的全新序列 (全空间搜索):")
    print(rec_df)

# ================= 8. 增强可视化：多维度景观分析 =================
import matplotlib.pyplot as plt

# 准备数据：获取每个序列的长度
all_lengths = np.array([len(s) for s in all_candidate_spacers])

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Global RBS Landscape Analysis: Multi-length & Multi-exploration', fontsize=20, y=0.95)

# --- 第一行：按长度 (6nt, 7nt, 8nt) 打印 μ vs σ ---
length_list = [6, 7, 8]
for i, L in enumerate(length_list):
    mask = (all_lengths == L)
    ax = axes[0, i]
    
    # 绘制散点
    scatter = ax.scatter(mu[mask], sigma[mask], c=mu[mask], cmap='viridis', s=2, alpha=0.4)
    ax.set_title(f'Sequence Length: {L}nt (N={sum(mask)})', fontsize=14)
    ax.set_xlabel('Predicted Mean (μ)', fontsize=12)
    ax.set_ylabel('Uncertainty (σ)', fontsize=12)
    
    # 添加各个长度下的 Top 1 标注
    best_idx_L = np.argmax(mu[mask])
    ax.annotate(f"Best {L}nt", 
                xy=(mu[mask][best_idx_L], sigma[mask][best_idx_L]),
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'))

# --- 第二行：不同探索度 (Kappa) 下的 UCB 得分分布 ---
# 我们观察 UCB = μ + κ * σ 的变化
kappas = [2, 10, 30] # 低探索(偏好稳定高产), 中探索, 高探索(寻找潜在奇迹)
for i, k in enumerate(kappas):
    ucb_val = mu + k * sigma
    ax = axes[1, i]
    
    # 绘制 UCB 得分与预测均值的关系
    scatter_u = ax.scatter(mu, ucb_val, c=sigma, cmap='plasma', s=2, alpha=0.3)
    ax.set_title(f'UCB Strategy (Kappa = {k})', fontsize=14)
    ax.set_xlabel('Predicted Mean (μ)', fontsize=12)
    ax.set_ylabel(f'UCB Score (μ + {k}σ)', fontsize=12)
    
    # 颜色条代表不确定性
    if i == 2:
        cbar = fig.colorbar(scatter_u, ax=ax)
        cbar.set_label('Uncertainty (σ)', rotation=270, labelpad=15)

# 调整布局
plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ================= 8. 可视化优化：全空间景观 + 策略选点对比 =================
# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial'] # 确保学术常用字体
plt.rcParams['axes.unicode_minus'] = False 

# 我们观察在三种代表性 Kappa 下，策略如何选择
kappas_pk = [2, 10, 30] 
y_max_val = y.max() # 用于 EI 计算的当前已知最大值

fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=300)
fig.suptitle('Acquisition Strategy Landscape: Discovery Map Across Exploration Intensities', 
             fontsize=20, fontweight='bold', y=1.05)

# 定义策略选点颜色 (区分是哪种策略选的点)
# 绿色代表 Greedy (保守)，红色代表当前 Kappa 下的 UCB (进取)，紫色代表 EI (均衡)
strategy_colors = {'Greedy': '#2CA02C', 'UCB_k': '#D62728', 'EI': '#9467BD'}

for i, k in enumerate(kappas_pk):
    ax = axes[i]
    
    # 当前 Kappa 下的 UCB 得分
    ucb_val_current = mu + k * sigma
    
    # --- A. 绘制背景：全量序列的景观 (image_3bd0e2 风格) ---
    # X轴是 μ，Y轴是 UCB分数，颜色是 σ
    scatter_bg = ax.scatter(mu, ucb_val_current, c=sigma, cmap='plasma', 
                            s=1, alpha=0.15, zorder=1)
    
    # --- B. 计算三种策略在当前景观下的采样点 (各取 Top 10) ---
    # 1. Greedy (不考虑 σ，只看最高 μ)
    idx_greedy_top = np.argsort(mu)[-10:][::-1]
    
    # 2. 当前 Kappa 下的 UCB (兼顾 μ 和 σ)
    idx_ucb_top = np.argsort(ucb_val_current)[-10:][::-1]
    
    # 3. EI (期望改善)
    scores_ei = get_acquisition_scores(mu, sigma, y_max_val, strategy='EI')
    idx_ei_top = np.argsort(scores_ei)[-10:][::-1]
    
    # --- C. 【核心改动】在景观图中高亮标出不同策略找到的点 ---
    # 区分不同策略的点，不需要标序列名
    
    # 1. 标出 Greedy 的点 (圆形)
    ax.scatter(mu[idx_greedy_top], ucb_val_current[idx_greedy_top], 
               color=strategy_colors['Greedy'], s=80, marker='o', edgecolors='white', 
               linewidth=1, label='Selected by Greedy', zorder=10)
    
    # 2. 标出当前 UCB 的点 (星星)
    ax.scatter(mu[idx_ucb_top], ucb_val_current[idx_ucb_top], 
               color=strategy_colors['UCB_k'], s=100, marker='o', edgecolors='white', 
               linewidth=1, label=f'Selected by UCB ($\kappa$={k})', zorder=11)
    
    # 3. 标出 EI 的点 (菱形)
    ax.scatter(mu[idx_ei_top], ucb_val_current[idx_ei_top], 
               color=strategy_colors['EI'], s=80, marker='o', edgecolors='white', 
               linewidth=1, label='Selected by EI', zorder=9)

    # --- D. 细节装饰与图例 ---
    ax.set_title(f'Discovery Strategy Map ($\kappa$ = {k})', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Mean ($\mu$)', fontsize=13)
    
    # Y轴根据当前 Kappa 动态调整
    ax.set_ylabel(f'UCB Score ($\mu$ + {k}$\sigma$)', fontsize=13)
    
    # 限制坐标轴范围以聚焦高分区域
    ax.set_ylim(-0.1, max(ucb_val_current)*1.05)
    
    # 只有第三张图显示颜色条和总图例
    if i == 2:
        cbar = fig.colorbar(scatter_bg, ax=ax)
        cbar.set_label('Uncertainty ($\sigma$)', rotation=270, labelpad=15)
        
        # 将图例合并放置
        fig.legend(loc='lower center', ncol=3, fontsize=11, frameon=True, shadow=True, 
                   bbox_to_anchor=(0.5, -0.08))

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("Global_Landscape_Strategy_PK.png", bbox_inches='tight')
plt.show()

# 维持原代码的打印辅助验证逻辑
# ... (保留原代码末尾的各长度最佳序列简报打印)

# --- 打印各长度的最佳序列 (辅助验证) ---
print("\n" + "="*30)
print("📊 各长度最优序列简报 (Greedy):")
for L in [6, 7, 8]:
    mask = (all_lengths == L)
    best_idx = np.where(mask)[0][np.argmax(mu[mask])]
    print(f"{L}nt Best: {all_candidate_spacers[best_idx]} | μ: {mu[best_idx]:.4f} | σ: {sigma[best_idx]:.4f}")
print("="*30)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 设置学术绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=300, sharey=True)
fig.suptitle('Global Sequence Landscape Exploration: Greedy vs. UCB vs. EI', 
             fontsize=22, fontweight='bold', y=1.05)

# 定义策略配置
strategies = ['Greedy', 'UCB', 'EI']
colors = ['#2CA02C', '#D62728', '#9467BD'] # 绿色(保守), 红色(进取), 紫色(均衡)
y_max_val = y.max() # 当前已知最大值，用于 EI 计算

# 2. 循环绘制每个策略的子图
for i, strat in enumerate(strategies):
    ax = axes[i]
    
    # --- A. 绘制全量 8.6万条序列作为背景 ---
    # 使用散点图，设置极小的 s 和 alpha，营造类似 image_3bd0e2 的景观感
    background = ax.scatter(mu, sigma, c=mu, cmap='plasma', s=1, alpha=0.1, zorder=1)
    
    # --- B. 计算当前策略得分并选出 Top 5 ---
    # 调用您代码中的 get_acquisition_scores
    scores = get_acquisition_scores(mu, sigma, y_max_val, strategy=strat, kappa=10)
    top_idx = np.argsort(scores)[-5:][::-1]
    
    # --- C. 突出显示选中的 Top 5 点 ---
    ax.scatter(mu[top_idx], sigma[top_idx], 
               color=colors[i], s=150, edgecolors='white', linewidth=1.5, 
               label=f'Selected by {strat}', zorder=10)
    
    # --- D. 标注选中的序列名称 ---
    for idx in top_idx:
        ax.annotate(all_candidate_spacers[idx], 
                    xy=(mu[idx], sigma[idx]), 
                    xytext=(8, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[i], alpha=0.7))

    # 子图装饰
    ax.set_title(f'Strategy: {strat}', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Mean ($\mu$)', fontsize=14)
    if i == 0:
        ax.set_ylabel('Predictive Uncertainty ($\sigma$)', fontsize=14)
    
    # 添加逻辑解释文本
    desc = {
        'Greedy': "Pure Exploitation\nTargets max $\mu$",
        'UCB': "Optimistic Exploration\nBalances $\mu$ and $\sigma$",
        'EI': "Probabilistic Improvement\nExpectation > $y_{max}$"
    }
    ax.text(0.05, 0.92, desc[strat], transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontweight='semibold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 3. 全局颜色条与布局调整
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(background, cax=cbar_ax, label='Predicted Mean ($\mu$) Value')

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("Global_Acquisition_Comparison.png", bbox_inches='tight')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 定义物理边界 (基于 SHAP 核心特征)
# 这里使用你定义的分子量与 DG 阈值
MW_MIN, MW_MAX = 700,1000 
DG_MIN = -8               

mask_physical = (X_virtual['Total_MW'] >= MW_MIN) & \
                (X_virtual['Total_MW'] <= MW_MAX) & \
                (X_virtual['DG_total'] >= DG_MIN)

# 2. 绘图配置
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300, sharey=True)
y_max_val = y.max()

# 策略颜色配置
strategies_config = [
    {'name': 'Greedy', 'color': '#2CA02C', 'label': 'Greedy'},
    {'name': 'UCB',    'color': '#D62728', 'label': 'UCB ($\kappa$=10)'},
    {'name': 'EI',     'color': '#9467BD', 'label': 'EI'}
]

# ==================== 左图：Vanilla AL (全彩色景观) ====================
# 这里展示原始 8.6万条序列的完整预测分布
scatter_full = ax1.scatter(mu, sigma, c=mu, cmap='plasma', s=2, alpha=0.3, zorder=2)
ax1.set_title('Vanilla Active Learning\n(Full Search Space - All Viable to Model)', fontsize=16, fontweight='bold')

for config in strategies_config:
    # 无约束选点
    scores = get_acquisition_scores(mu, sigma, y_max_val, strategy=config['name'], kappa=10)
    top_idx = np.argsort(scores)[-8:][::-1]
    ax1.scatter(mu[top_idx], sigma[top_idx], color=config['color'], s=180, 
                edgecolors='white', linewidth=1.5, label=config['label'], zorder=10)

# ==================== 右图：GAL (物理过滤景观) ====================
# 符合条件的点保持彩色
ax2.scatter(mu[mask_physical], sigma[mask_physical], c=mu[mask_physical], 
            cmap='plasma', s=2, alpha=0.3, zorder=2)
# 被刨除的点变为灰色
ax2.scatter(mu[~mask_physical], sigma[~mask_physical], c='lightgray', 
            s=1, alpha=0.1, zorder=1, label='Physically Excluded')

ax2.set_title('Guided Active Learning (GAL)\n(Filtered by Molecular Weight & $\Delta G$)', fontsize=16, fontweight='bold')

for config in strategies_config:
    # GAL 选点：只在 mask_physical 为 True 的范围内选
    mu_filtered = mu.copy()
    mu_filtered[~mask_physical] = -np.inf 
    scores = get_acquisition_scores(mu_filtered, sigma, y_max_val, strategy=config['name'], kappa=10)
    top_idx = np.argsort(scores)[-8:][::-1]
    ax2.scatter(mu[top_idx], sigma[top_idx], color=config['color'], s=180, 
                edgecolors='white', linewidth=1.5, label=f"GAL-{config['label']}", zorder=10)

# 3. 细节统一修饰
for ax in [ax1, ax2]:
    ax.set_xlabel('Predicted Mean ($\mu$)', fontsize=13)
    ax.legend(loc='upper left', frameon=True, shadow=True)

ax1.set_ylabel('Predictive Uncertainty ($\sigma$)', fontsize=13)

# 添加全局颜色条
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(scatter_full, cax=cbar_ax, label='Predicted Expression Value ($\mu$)')

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("Gal.png", bbox_inches='tight')
plt.show()