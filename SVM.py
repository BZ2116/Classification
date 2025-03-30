"""
author:Bruce Zhao
date: 2025/3/26 10:42
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#SVM 对特征缩放敏感，建议对数据进行标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#kernel 参数指定核函数类型，
# 常用的核函数包括 'linear'（线性核）、'poly'（多项式核）和 'rbf'（径向基函数）。
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, y_train)

#模型预测
y_pred = clf.predict(X_test)

#计算模型的准确率，并输出分类报告
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")
print("分类报告:")
print(classification_report(y_test, y_pred))

