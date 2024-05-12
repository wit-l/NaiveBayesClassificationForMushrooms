#!/opt/anaconda3/envs/sklearn/bin/python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 加载数据
data = pd.read_csv("mushrooms.csv")

# print(type(data))
# print(data)
# print(data.columns[1:])

# 编码
les = {}
for col in data.columns[1:]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    les[col] = le

# print(type(data))
# print(data)

# 'class'是目标列
X = data.drop("class", axis=1)
y = data["class"].map({"p": 1, "e": 0})  # type:ignore

# print(X)
# print(type(X), X.shape)
# print(y)
# print(type(y), y.shape)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建高斯朴素贝叶斯分类器
gnb = GaussianNB(var_smoothing=2.848035868435802e-4)
# var_smoothing=2.848035868435802e-4
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print("Accuracy of Gaussian Naive Bayes:", accuracy_gnb)

# 创建多项式朴素贝叶斯分类器
mnb = MultinomialNB(alpha=0.01)
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
print("Accuracy of Multinomial Naive Bayes:", accuracy_mnb)

# 创建神经网络分类器
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("Accuracy of MLP Classifier:", accuracy_mlp)

print("\n超参数调优")
print("\n高斯朴素贝叶斯模型")
param_grid = {"var_smoothing": np.logspace(0, -9, num=100)}
gnb_ = GaussianNB()
grid_search_gnb = GridSearchCV(gnb_, param_grid, cv=5)
grid_search_gnb.fit(X_train, y_train)
print("最优参数：", grid_search_gnb.best_params_)
print("最佳交叉验证得分：{:.2f}".format(grid_search_gnb.best_score_))

print("\n多项式朴素贝叶斯模型")
param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0]}
mnb_ = MultinomialNB()
grid_search_mnb = GridSearchCV(mnb_, param_grid, cv=5, scoring="accuracy")
grid_search_mnb.fit(X_train, y_train)
print("最优参数：", grid_search_mnb.best_params_)
print("最佳交叉验证得分：{:.2f}".format(grid_search_mnb.best_score_))

while True:
    print("\n-----------通过手动输入样本特征测试个模型的输出结果---------------")
    user_input = input("输入蘑菇样本特征值：")
    X_t = pd.DataFrame([user_input.split(",")], columns=X.columns)  # type: ignore
    for col in X_t.columns:
        X_t[col] = les[col].transform(X_t[col])

    print("\n高斯朴素贝叶斯模型")
    y_pred = gnb.predict(X_t)[0]
    if y_pred == 1:
        result = "有毒"
    else:
        result = "无毒"
    print("预测结果：", result)

    print("\n多项式朴素贝叶斯模型")
    y_pred = mnb.predict(X_t)[0]
    if y_pred == 1:
        result = "有毒"
    else:
        result = "无毒"
    print("预测结果：", result)

    print("\n神经网络模型")
    y_pred = mlp.predict(X_t)[0]
    if y_pred == 1:
        result = "有毒"
    else:
        result = "无毒"
    print("预测结果：", result)
    is_continue = input("是否继续预测？（y/n)：")
    if is_continue == "y" or is_continue == "Y":
        continue
    else:
        break
