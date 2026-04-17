import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import griddata
# 1. 生成模拟的复杂健身景观 (Fitness Landscape)

def fitness_function(x, y):

    # 使用多个正弦波叠加模拟充满噪声和局部极值的景观

    z = np.sin(0.1*x) * np.cos(0.1*y) + \
         0.5 * np.cos(0.3*x) * np.sin(0.5*y) + \
         0.2 * np.random.randn(*x.shape) # 加入随机噪声

    return z



x = np.linspace(-20, 20, 100)

y = np.linspace(-20, 20, 100)

X, Y = np.meshgrid(x, y)

Z = fitness_function(X, Y)



# 2. 模拟局部采样点 (N=200)

np.random.seed(42)

sample_x = np.random.uniform(-10, 10, 100) # 采样集中在某一区域

sample_y = np.random.uniform(-10, 10, 100)

sample_z = fitness_function(sample_x, sample_y)



# 3. 设置绘图

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(111, projection='3d')



# 绘制全局背景景观 (半透明灰，表示未探索区域)

#surf = ax.plot_surface(X, Y, Z, cmap='Greys', alpha=0.3, antialiased=True, shade=True)



# 绘制你的采样点 (带颜色的散点，表示已观测数据)

sc = ax.scatter(sample_x, sample_y, sample_z, c=sample_z, cmap='viridis', s=20, alpha=0.8, label='Initial Samples (N=200)')



# 4. 模拟 UCB 引导的探索方向 (箭头从采样区指向景观高点)

# 选取几个采样点作为起点，指向高值区域

#ax.quiver(-5, -5, 0.5, 10, 12, 1.5, color='red', lw=3, arrow_length_ratio=0.1, label='Guided Extrapolation (UCB)')

#ax.quiver(0, 2, 0.2, -8, 15, 1.2, color='red', lw=2, arrow_length_ratio=0.1)



# 5. 修饰图表

ax.set_title("Guided Active Learning: Local Sampling to Global Exploration", fontsize=15)
# 创建一个精细的网格用于绘制拟合面
xi = np.linspace(min(sample_x), max(sample_x), 50)
yi = np.linspace(min(sample_y), max(sample_y), 50)
XI, YI = np.meshgrid(xi, yi)

# 使用立方插值 (cubic) 来模拟模型学习到的平滑规律
# 这比线性平面更能体现生物景观的非线性
#from scipy.interpolate import Rbf
#rbf = Rbf(sample_x, sample_y, sample_z, function='multiquadric', smooth=2.0)
#XI, YI = np.meshgrid(xi, yi)
ZI_fitted = griddata((sample_x, sample_y), sample_z, (XI, YI), method='cubic')
# B. 绘制拟合出的局部曲面 (代表模型的认知层)
# 使用 'Spectral' 或 'viridis' 增加色彩感，alpha 设为 0.7 保持半透明
#surf = ax.plot_surface(XI, YI, ZI_fitted, cmap='Spectral_r', alpha=0.7, 
#                        antialiased=True, shade=True, edgecolor='none', zorder=2)

# C. 绘制原始采样点 (置于曲面上方，体现“真实数据”与“拟合规律”的关系)
#ax.scatter(sample_x, sample_y, sample_z, c='black', s=15, alpha=0.5, label='Raw Observations', zorder=5)



# 调整视角以突出景观波动

ax.view_init(elev=35, azim=-45)

# 1. 完全隐藏坐标轴、刻度和背景网格

ax.set_axis_off()

plt.legend()

plt.tight_layout()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_distribution_comparison():
    # 1. 设置背景景观 (真实地形)
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(x, y)
    # 真实景观：有一个显著的高峰和复杂的噪声背景
    Z_truth = 2.5 * np.exp(-((X-12)**2 + (Y-12)**2)/60) + 0.3 * np.sin(0.8*X) * np.cos(0.8*Y)

    np.random.seed(42)

    # 2. 生成 Traditional AL 数据点 (特点：样本多，分布广，无序)
    # 模拟传统方法为了测准全景观，在各处都撒了点
    n_al = 800  # 强调数据量多
    al_x = np.random.uniform(-20, 20, n_al)
    al_y = np.random.uniform(-20, 20, n_al)
    al_z = 2.5 * np.exp(-((al_x-12)**2 + (al_y-12)**2)/60) + 0.3 * np.sin(0.8*al_x) * np.cos(0.8*al_y) + np.random.normal(0, 0.15, n_al)

    # 3. 生成 GAL 数据点 (特点：样本极少，初期在低位，后期精准外推)
    n_gal_init = 100   # 初期随机采样（低值区为主）
    n_gal_guided = 100 # 后期策略引导（向高值区探索）
    
    # 初始点集中在远离高峰的区域（体现“起步难”）
    gal_init_x = np.random.uniform(-18, 0, n_gal_init)
    gal_init_y = np.random.uniform(-18, 0, n_gal_init)
    
    # 引导点沿着一条狭窄的、符合机理的路径向高峰渗透
    gal_guided_x = np.linspace(0, 15, n_gal_guided) + np.random.normal(0, 1.5, n_gal_guided)
    gal_guided_y = np.linspace(0, 15, n_gal_guided) + np.random.normal(0, 1.5, n_gal_guided)
    
    gal_x = np.concatenate([gal_init_x, gal_guided_x])
    gal_y = np.concatenate([gal_init_y, gal_guided_y])
    gal_z = 2.5 * np.exp(-((gal_x-12)**2 + (gal_y-12)**2)/60) + 0.3 * np.sin(0.8*gal_x) * np.cos(0.8*gal_y) + np.random.normal(0, 0.05, 200)

    # 4. 绘图
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    
    # --- 子图 1: Traditional AL (暴力搜索/大数据) ---
    ax1 = fig.add_subplot(121, projection='3d')
    # 景观背景
    ax1.plot_surface(X, Y, Z_truth, cmap='Greys', alpha=0.1)
    # 散点：颜色浅、点多且杂乱
    ax1.scatter(al_x, al_y, al_z, c='gray', s=10, alpha=0.3, label='Global Brute-force (N=800)')
    ax1.set_title("Traditional AL: High-Cost & Low-Efficiency", fontsize=14)
    ax1.set_axis_off()
    ax1.view_init(30, -60)
    ax1.legend(loc='lower left')

    # --- 子图 2: Guided AL (GAL) (策略搜索/少样本) ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z_truth, cmap='Greys', alpha=0.1)
    # 初始点（灰色，低位）
    ax2.scatter(gal_init_x, gal_init_y, gal_z[:100], c='gray', s=20, alpha=0.4, label='Initial Exploration')
    # 引导点（彩色，体现突破）
    ax2.scatter(gal_guided_x, gal_guided_y, gal_z[100:], c=gal_z[100:], cmap='viridis', s=40, alpha=0.9, edgecolors='black', linewidth=0.3, label='Guided Discovery (N=100)')
    ax2.set_title("Guided AL: Data-Efficient Navigation", fontsize=14)
    ax2.set_axis_off()
    ax2.view_init(30, -60)
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_distribution_comparison()

    import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 模拟生物景观 (Fitness Landscape)
def landscape(x, y):
    # 一个位于 (12, 12) 的全局高峰
    peak = 2.5 * np.exp(-((x - 12)**2 + (y - 12)**2) / 60)
    # 背景噪声和局部波动
    noise = 0.3 * np.sin(0.6 * x) * np.cos(0.6 * y)
    return peak + noise

# 生成网格背景
x_range = np.linspace(-20, 20, 100)
y_range = np.linspace(-20, 20, 100)
X, Y = np.meshgrid(x_range, y_range)
Z_bg = landscape(X, Y)

# 2. 生成数据集点
np.random.seed(42)

# --- 数据集 A: 传统 AL (数据量大, 取值分散, 覆盖广) ---
n_al = 600
al_x = np.random.uniform(-20, 20, n_al)
al_y = np.random.uniform(-20, 20, n_al)
al_z = landscape(al_x, al_y) + np.random.normal(0, 0.1, n_al)

# --- 数据集 B: 你的 GAL (数据量少, 包含低值区, 且能精准爬坡) ---
n_gal_init = 80    # 初始随机点 (多处于低值区)
n_gal_guided = 70  # 引导探索点 (沿路径向高值区集中)

# 初始点集中在远离高峰的左下区域
gal_init_x = np.random.uniform(-18, 0, n_gal_init)
gal_init_y = np.random.uniform(-18, 0, n_gal_init)

# 引导点：模拟模型发现规律后，顺着物理梯度向(12, 12)外推
gal_guided_x = np.linspace(0, 14, n_gal_guided) + np.random.normal(0, 1.5, n_gal_guided)
gal_guided_y = np.linspace(0, 14, n_gal_guided) + np.random.normal(0, 1.5, n_gal_guided)

gal_x = np.concatenate([gal_init_x, gal_guided_x])
gal_y = np.concatenate([gal_init_y, gal_guided_y])
gal_z = landscape(gal_x, gal_y) + np.random.normal(0, 0.05, len(gal_x))

# 3. 绘图
fig = plt.figure(figsize=(18, 9), facecolor='white')

# --- 子图 1: 传统 AL 数据分布 ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_bg, cmap='Greys', alpha=0.1, antialiased=True) # 背景
ax1.scatter(al_x, al_y, al_z, c='gray', s=10, alpha=0.3, label=f'Traditional AL (N={n_al})')
ax1.set_title("Traditional AL: High-Cost & Sparse Utility", fontsize=14)
ax1.set_xlabel('Sequence Space X')
ax1.set_ylabel('Sequence Space Y')
ax1.set_zlabel('Expression Level (Z)')
ax1.view_init(30, -60)
ax1.legend()

# --- 子图 2: GAL 数据分布 ---
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_bg, cmap='Greys', alpha=0.1, antialiased=True) # 背景
# 初始点（深灰色）
ax2.scatter(gal_init_x, gal_init_y, gal_z[:n_gal_init], c='#555555', s=25, alpha=0.6, label='Initial Random Samples')
# 引导点（彩色，体现爬坡过程）
sc = ax2.scatter(gal_guided_x, gal_guided_y, gal_z[n_gal_init:], 
                c=gal_z[n_gal_init:], cmap='viridis', s=45, alpha=0.9, 
                edgecolors='black', linewidth=0.3, label='Guided Active Samples')
ax2.set_title("Guided AL (GAL): Data-Efficient Discovery", fontsize=14)
ax2.set_xlabel('Sequence Space X')
ax2.set_ylabel('Sequence Space Y')
ax2.set_zlabel('Expression Level (Z)')
ax2.view_init(30, -60)
ax2.legend()

plt.tight_layout()
plt.show()

