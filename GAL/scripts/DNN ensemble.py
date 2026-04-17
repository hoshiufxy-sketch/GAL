from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor # 用于辅助 RFE

# ... [保留你之前的特征提取和数据读取部分] ...
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
df = pd.read_csv('GAL/data/spacer_onehot_features.csv', header=0,nrows=210)## nrows=99 194 202
print(df.head())  # 查看前几行
print(df.columns)  # 检查列名
# 提取输入特征（Z列到BH列）
X = df.iloc[:, 2:] # 25列，BH列是第60列 去掉values!!!16380

# 提取输出值（C列）
y = df.iloc[:, 1] # C列是第2列
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #42 21 0

# 提取特征与标签
X = df.iloc[:, 2:] 
y = df.iloc[:, 1] 

# 1. 必须进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 初始化 XGBoost 回归模型
model = xgb.XGBRegressor(n_estimators=300,learning_rate=0.05, gamma=0.8, max_depth=10,random_state=42,)

# 训练模型
model.fit(X_train, y_train)

# 执行 RFE 特征选择（选前n个特征，可根据需要修改）
selector = RFE(model, n_features_to_select=600, step=100)
selector = selector.fit(X, y)

selected_feature_names = X.columns[selector.get_support()]
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# model.fit(X_selected, y)

# ================= 5. Boost Ensemble 训练逻辑 =================
n_ensemble = 5  # 集成模型的数量，建议 5-10 个
ensemble_models = []

print(f"正在训练 {n_ensemble} 个集成模型...")

# 这里的 X_train_selected 是经过 RFE 筛选后的特征
for i in range(n_ensemble):
    # Bootstrap 抽样：有放回地抽取与原训练集一样多的样本
    # random_state=i 保证每个子模型看到的样本组合不同
    # 使用 Bootstrap 抽样
    indices = np.random.choice(len(X_train_selected), len(X_train_selected), replace=True)
    X_resample = X_train_selected[indices]
    y_resample = y_train.iloc[indices]
    
    # 定义 DNN (MLPRegressor)
    # hidden_layer_sizes 定义了神经元层级
    sub_model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16), 
        activation='relu', 
        solver='adam', 
        alpha=0.005,      # L2 正则化防止过拟合
        max_iter=1000,    # 迭代次数
        random_state=i,
        early_stopping=True # 自动划分验证集，防止过拟合
    )
    
    sub_model.fit(X_resample, y_resample)
    ensemble_models.append(sub_model)

print("集成模型训练完成。")

importances = model.feature_importances_


# 在测试集上进行预测
y_pred = model.predict(X_test)


# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f'R^2:{r2:.6f}')
from scipy.stats import norm
# ================= 6. 集成预测函数 =================
def predict_with_ensemble(models, X_data):
    all_preds = np.array([m.predict(X_data) for m in models])
    mu = np.mean(all_preds, axis=0)
    sigma = np.std(all_preds, axis=0)
    return mu, sigma
# 在测试集上评估
y_pred_mu, y_uncertainty = predict_with_ensemble(ensemble_models, X_test_selected)

# 计算指标
r2 = r2_score(y_test, y_pred_mu)
print(f"Ensemble R^2: {r2:.6f}")


# ================= 9. SHAP 解释性分析模块 =================
# ================= 9. 全量测试集 SHAP 深度解释模块 =================
import shap

def run_full_test_shap(ensemble_models, X_test, selected_feature_names):
    print("\n[SHAP] 正在为全部测试集（N={}) 生成机理说明图...".format(len(X_test)))
    
    # 1. 特征对齐：确保测试集只包含 RFE 选出的 30/600 个特征
    X_test_selected = X_test[selected_feature_names]
    
    # 2. 获取集成模型中第一个子模型作为解释对象 (或者计算平均 SHAP，此处以单模型为例)
    # XGBoost 的 TreeExplainer 是目前解释非线性生物效应最准的工具
    explainer = shap.TreeExplainer(ensemble_models[0])
    shap_values = explainer.shap_values(X_test_selected)
    expected_value = explainer.expected_value

    # --- A. 全量 Summary Plot (条形图 + 散点图) ---
    # 这种图能一眼看出哪些物理指标（如 atg_acc）是正向增益
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_selected, plot_type="dot", show=False)

    plt.tight_layout()
    plt.savefig("GAL_SHAP_Summary_Full.png", dpi=300)
    plt.show()

    # --- B. 全量 Decision Plot (决策路径图) ---
    # 这张图会显示测试集里所有序列从基准线到最终预测值的“心路历程”
    # 如果线条大多向右偏移，说明这些序列共同具备某些高表达特征
    plt.figure(figsize=(12, 8))
    # 注意：如果测试集样本过多（>500），建议只画 Top 50 或进行平滑处理
    # 这里展示全部测试集
    shap.decision_plot(expected_value, shap_values, 
                       features=X_test_selected, 
                       feature_names=list(selected_feature_names),
                       link='identity', 
                       show=False)

    plt.tight_layout()
    plt.savefig("GAL_SHAP_Decision_Full.png", dpi=300)
    plt.show()

    # --- C. Heatmap Plot (热图) ---
    # 这对于发现特征之间的协同效应（Co-regulation）非常有帮助
    plt.figure(figsize=(12, 6))
    shap.plots.heatmap(shap.Explanation(values=shap_values, 
                                       base_values=expected_value, 
                                       data=X_test_selected.values, 
                                       feature_names=selected_feature_names,), 
                      show=False)

    plt.tight_layout()
    plt.show()

# 执行分析
# run_full_test_shap(ensemble_models, X_test, selected_feature_names)
