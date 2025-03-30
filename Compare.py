import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np

# ================= 环境设置 =================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ================= 数据准备 =================
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# ================= 模型训练 =================
models = {
    '朴素贝叶斯': GaussianNB(),
    'SVM': SVC(probability=True),
    '决策树': DecisionTreeClassifier(max_depth=3)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = classification_report(y_test, y_pred,
                                           target_names=iris.target_names,
                                           output_dict=True)

# ================= 数据输出 =================
print("\n" + "="*40 + " 详细性能数据 " + "="*40)
for model_name, report in results.items():
    print(f"\n{model_name} 分类报告：")
    df_report = pd.DataFrame(report).transpose()
    print(df_report.round(2).to_string())

# ================= 可视化设计 =================
# 准备可视化数据
def create_metric_df(reports):
    data = []
    for model, report in reports.items():
        for class_name in iris.target_names:
            data.append({
                '模型': model,
                '类别': class_name,
                'F1值': report[class_name]['f1-score']
            })
        data.append({
            '模型': model,
            '类别': '宏平均',
            'F1值': report['macro avg']['f1-score']
        })
    return pd.DataFrame(data)

metric_df = create_metric_df(results)

# 创建颜色映射
colors = plt.cm.tab10.colors[:4]  # 4个颜色对应4个类别

# ====== 修正后的条形图 ======
plt.figure(figsize=(12, 6), dpi=100)
models_order = ['朴素贝叶斯', 'SVM', '决策树']
categories = iris.target_names.tolist() + ['宏平均']

n_models = len(models_order)
bar_width = 0.15
spacing = 0.05

# 生成位置矩阵（保持原有计算逻辑）
x_base = np.arange(n_models)
x_positions = []
for model_idx in range(n_models):
    start = x_base[model_idx] - (bar_width*(len(categories)/2) + spacing*(len(categories)-1)/2)
    for cat_idx in range(len(categories)):
        x = start + cat_idx*(bar_width + spacing)
        x_positions.append(x)

# 提取F1值并排序
f1_values = []
for model in models_order:
    for category in categories:
        f1 = metric_df[(metric_df['模型'] == model) &
                      (metric_df['类别'] == category)]['F1值'].values[0]
        f1_values.append(f1)

# 修正颜色参数
bar_colors = colors * n_models  # 正确重复颜色序列

bars = plt.bar(x_positions, f1_values, width=bar_width,
              color=bar_colors,  # 使用列表乘法生成的颜色序列
              edgecolor='k', linewidth=0.5)


# 添加数据标签
for x, y in zip(x_positions, f1_values):
    plt.text(x, y + 0.01, f'{y:.2f}',
             ha='center', va='bottom',
             fontsize=9, rotation=45)


# 自定义图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=category)
                   for i, category in enumerate(categories)]
plt.legend(handles=legend_elements,
           bbox_to_anchor=(1.02, 1),
           loc='upper left')

# ====== 热力图 ======
plt.figure(figsize=(10, 6), dpi=100)
heatmap_data = metric_df.pivot(index='模型', columns='类别', values='F1值')

# 创建矩阵数据
matrix = heatmap_data.values
rows = heatmap_data.index.tolist()
cols = heatmap_data.columns.tolist()

# 绘制热力图
im = plt.imshow(matrix, cmap='YlGnBu', aspect='auto')

# 添加颜色条
cbar = plt.colorbar(im, shrink=0.8)
cbar.set_label('F1值', rotation=270, labelpad=15)

# 坐标轴设置
plt.xticks(np.arange(len(cols)), cols, rotation=45)
plt.yticks(np.arange(len(rows)), rows)
plt.xlabel('')
plt.ylabel('')

# 添加数值标注
for i in range(len(rows)):
    for j in range(len(cols)):
        text = f'{matrix[i, j]:.2f}'
        plt.text(j, i, text,
                 ha='center', va='center',
                 color='black', fontsize=10)

plt.title('分类器F1值热力图对比', fontsize=14, pad=20)
plt.tight_layout()

plt.show()
