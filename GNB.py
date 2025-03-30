"""
author:Bruce Zhao
date: 2025/3/26 9:34
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class_prior = [0.1, 0.3, 0.6]  # 假设类别 0、1、2 的先验概率分别为 0.1、0.3、0.6
gnb = GaussianNB(priors=class_prior)
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")
