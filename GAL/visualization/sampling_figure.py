import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 设置随机种子
np.random.seed(123)

# 1. 定义复杂健身景观函数（含高频噪声）
def fitness_landscape(x, y):
    # 基础景观
    z = np.sin(0.15*x) * np.cos(0.15*y) + 0.5 * np.cos(0.5*x) * np.sin(0.5*y)
    # 真实的目标高峰
    target_peak = 2.0 * np.exp(-((x-15)**2 + (y-15)**2)/30)
    # 显著的实验噪声
    noise = 0.2 * np.random.randn(*x.shape)
    return z + target_peak + noise

# 生成背景网格
x_range = np.linspace(-20, 20, 50)
y_range = np.linspace(-20, 20, 50)
X, Y = np.meshgrid(x_range, y_range)
X_flat = np.vstack([X.ravel(), Y.ravel()]).T
Z_global = fitness_landscape(X, Y)

# 2. 初始随机采样点（N=10，集中在左下区域）
train_indices = np.random.choice(len(X_flat), size=10, replace=False)
train_x = X_flat[train_indices]
train_y = fitness_landscape(train_x[:, 0], train_x[:, 1])

# 3. 拟合高斯过程模型
# 使用高斯过程来估计均值和不确定性（标准差）
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.1**2)
gp.fit(train_x, train_y)

# 4. 预测全空间
mu, sigma = gp.predict(X_flat, return_std=True)
MU = mu.reshape(X.shape)
SIGMA = sigma.reshape(X.shape)

# 5. 模拟两种采样函数
# A. AL采样：选择不确定性最高的点（sigma最大）
next_al_index = np.argmax(sigma)
next_al_x = X_flat[next_al_index]
next_al_z = fitness_landscape(next_al_x[0], next_al_x[1])

# B. GAL采样：选择UCB最高的点（mu + kappa*sigma, kappa=2）
kappa = 2.0
ucb = mu + kappa * sigma
next_gal_index = np.argmax(ucb)
next_gal_x = X_flat[next_gal_index]
next_gal_z = fitness_landscape(next_gal_x[0], next_gal_x[1])

# 6. 绘图对比
fig = plt.figure(figsize=(16, 10), facecolor='white')

# 子图1：AL 采样结果 (追求全局重建)
ax1 = fig.add_subplot(121, projection='3d')
# 绘制真实的嘈杂背景
ax1.plot_surface(X, Y, Z_global, cmap='Greys', alpha=0.15, antialiased=True, shade=True)
# 绘制拟合的模型曲面
ax1.plot_surface(X, Y, MU, cmap='YlGnBu', alpha=0.5, antialiased=True, shade=False)
# 绘制初始采样点
ax1.scatter(train_x[:, 0], train_x[:, 1], train_y, c='black', s=40, alpha=0.8, label='Initial Samples')
# 绘制 AL 下一步采样点（彩色，散布在灰色背景上）
ax1.scatter(next_al_x[0], next_al_x[1], next_al_z, c='#FF8C00', s=100, edgecolors='white', lw=2, label='AL Next Sample (Uncertainty)')
# 视觉修饰
ax1.set_axis_off()
ax1.view_init(elev=35, azim=-60)
ax1.set_title("TAL (AL): Uncertainty Sampling for Landscape Reconstruction", fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# 子图2：GAL 采样结果 (追求性能外推)
ax2 = fig.add_subplot(122, projection='3d')
# 绘制真实的嘈杂背景
ax2.plot_surface(X, Y, Z_global, cmap='Greys', alpha=0.15, antialiased=True, shade=True)
# 绘制拟合的模型曲面
ax2.plot_surface(X, Y, MU, cmap='Spectral_r', alpha=0.6, antialiased=True, shade=False)
# 绘制初始采样点
ax2.scatter(train_x[:, 0], train_x[:, 1], train_y, c='black', s=40, alpha=0.8, label='Initial Samples')
# 绘制 GAL 下一步采样点（红色，指向全局高点区域）
ax2.scatter(next_gal_x[0], next_gal_x[1], next_gal_z, c='red', s=100, edgecolors='white', lw=2, label='GAL Next Sample (UCB)')
# 视觉修饰
ax2.set_axis_off()
ax2.view_init(elev=35, azim=-60)
ax2.set_title("GAL: Directed Extrapolation towards Peak (UCB)", fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

plt.tight_layout()
plt.show()