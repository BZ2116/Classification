"""
author:Bruce Zhao
date: 2025/3/30 11:02
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------- 关键修复：设置中文字体 ----------- #
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体（如黑体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 1. 加载数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = [name.replace(" (cm)", "") for name in iris.feature_names]  # 简化特征名称
class_names = iris.target_names

# 转换为DataFrame方便查看
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = df['target'].map({0: '山鸢尾', 1: '变色鸢尾', 2: '维吉尼亚鸢尾'})  # 使用中文类别名

print("数据集样例:")
print(df.head())

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 创建并训练决策树模型
model = DecisionTreeClassifier(
    max_depth=3,   # 限制树深度便于可视化
    criterion='gini',
    random_state=42
)
model.fit(X_train, y_train)

# 4. 预测与评估
y_pred = model.predict(X_test)

print("\n模型评估:")
print(f"准确率: {accuracy_score(y_test, y_pred):.2%}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 5. 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    filled=True,
    feature_names=feature_names,
    class_names=class_names,
    rounded=True,
    proportion=True,
    fontsize=10
)
plt.title("鸢尾花分类决策树可视化", fontsize=18)
plt.show()

# 6. 特征重要性分析
plt.figure(figsize=(10, 6))
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

plt.barh(
    range(X.shape[1]),
    importance[indices],
    align='center',
    color='skyblue'
)
plt.yticks(range(X.shape[1]), [feature_names[i] for i in indices])
plt.xlabel("特征重要性", fontsize=12)
plt.title("鸢尾花分类特征重要性排名", fontsize=15)
plt.gca().invert_yaxis()  # 重要性从高到低显示
plt.show()