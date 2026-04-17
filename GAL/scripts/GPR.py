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
df = pd.read_csv('GAL/data/spacer_ViennaRNA_features.csv', header=0,nrows=210)## nrows=99 194 202
print(df.head())  # 查看前几行
print(df.columns)  # 检查列名
# 提取输入特征（Z列到BH列）
X = df.iloc[:, 2:] # 25列，BH列是第60列 去掉values!!!16380

# 提取输出值（C列）
y = df.iloc[:, 1] # C列是第2列
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #42 21 0

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.ensemble import RandomForestRegressor # 用于 RFE 代理筛选

# ================= 4. Gaussian Process 模型配置 =================

# 定义核函数：常数项 * RBF核 + 噪声核
kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2)) + WhiteKernel(noise_level=0.1)

# 初始化 GP 模型
# n_restarts_optimizer 设为 10 以上以防止陷入局部最优解
model = GaussianProcessRegressor(
        kernel=C(1.0, (1e-3, 1e6)) * RBF(1.0, (1e-1, 1e2)) + WhiteKernel(noise_level=0.1),
        n_restarts_optimizer=20, 
        alpha=0.1, 
        random_state=42
)
# 训练模型
model.fit(X_train, y_train)

# 执行 RFE 特征选择（选前n个特征，可根据需要修改）
selector = RFE(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42), 
    n_features_to_select=600, 
    step=100
)
selector = selector.fit(X, y)

# 取被选中的特征名称
selected_feature_names = X.columns[selector.get_support()]

# 重新拟合模型以获取这些特征的 importance（用筛选后的特征）
X_selected = X[selected_feature_names]


# ================= 5. Boost Ensemble 训练逻辑 =================
n_ensemble = 5  # 集成模型的数量，建议 5-10 个
ensemble_models = []

print(f"正在训练 {n_ensemble} 个集成模型...")



print("集成模型训练完成。")

# 在测试集上进行预测
y_pred = model.predict(X_test)


# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f'R^2:{r2:.6f}')
model.fit(X_selected, y)
from scipy.stats import norm
# ================= 6. 集成预测函数 =================
def predict_with_gp(model, X_data):
    # return_std=True 即可获得预测均值和标准差
    mu, sigma = model.predict(X_data, return_std=True)
    return mu, sigma

# 在测试集上评估
y_pred_mu, y_uncertainty = predict_with_gp(model, X_test)

# 计算指标
r2 = r2_score(y_test, y_pred_mu)
mse = mean_squared_error(y_test, y_pred_mu)
print(f'GP Mean Squared Error: {mse:.6f}')
print(f'GP R^2: {r2:.6f}')
