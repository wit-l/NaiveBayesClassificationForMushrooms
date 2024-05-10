#!/opt/anaconda3/envs/sklearn/bin/python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 加载数据
data = pd.read_csv("mushrooms.csv")

# print(type(data))
# print(data)

# 编码
le = LabelEncoder()
data = data.apply(le.fit_transform)

# print(type(data))
# print(data)

# 假设数据已经清洗完成，'target'是目标列
X = data.drop("class", axis=1)
y = data["class"]
# print(X)
# print(type(X), X.shape)
# print(y)
# print(type(y), y.shape)


# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X, y)
print(X_scaled)
print(X_scaled.shape)
print(type(X_scaled))

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 创建高斯朴素贝叶斯分类器
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print("Accuracy of Gaussian Naive Bayes:", accuracy_gnb)

# 创建神经网络分类器
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("Accuracy of MLP Classifier:", accuracy_mlp)
