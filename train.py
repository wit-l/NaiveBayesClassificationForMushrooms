import pandas as pd
import numpy as np
from IPython.display import display

# LabelEncoder 用于标签编码 例['red', 'green', 'blue'] => [0, 1, 2]
from sklearn.preprocessing import LabelEncoder

# train_test_split 分割训练集和测试集
from sklearn.model_selection import train_test_split

# 高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

data = pd.read_csv("./mushrooms.csv")
print("读入csv后，存放samples的data中前5行内容：")
display(data.head())

print("类别：\n", data["class"].unique())

print("各类别的数量")
# print(data['class'].value_counts())
print(data.groupby("class").size())

labelEncoder = LabelEncoder()
data = data.apply(labelEncoder.fit_transform)
print("对属性值和类别编码后得到的数据：")
display(data.head())

# samples
X = data.iloc[:, 1:]  # DataFrame
# Labels
y = data.iloc[:, 0]  # Series

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
nb = MultinomialNB()
# training
nb.fit(X_train, y_train)

y_predict = nb.predict(X_test)
cm = confusion_matrix(y_test, y_predict)

print("前四十五个测试数据的真实类别编号：")
print(np.array(y_test)[:45])

print("\n前四十五个测试数据的预测类别编号：")
print(y_predict[:45])

print("confusion matrix:\n", cm)
print("混淆矩阵类型:", type(cm))

tp = cm[0, 0]  # 真阳
fp = cm[0, 1]  # 假阳
fn = cm[1, 0]  # 假阴
tn = cm[1, 1]  # 真阴

# print("accuracy_score:", accuracy_score(y_test, y_predict))
# print("precision_score:", precision_score(y_test, y_predict))
# print("recall_score:", recall_score(y_test, y_predict))

accuracy_rate = (tp + tn) / np.sum(cm)
precision_rate = tp / np.sum(cm, 1)[0]
recall_rate = tp / np.sum(cm, 0)[0]

print("准确率", accuracy_rate)
print("精确率", precision_rate)
print("召回率", recall_rate)

print(classification_report(y_test, y_predict, target_names=["无毒", "有毒"]))


# print(
#     "cap-shape,cap-surface,cap-color,bruises,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring,stalk-surface-below-ring,stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat"
# )
# user_in_str = input(
#     "请输入蘑菇的各个特征值(按照以上特征的顺序输入，并用','分隔各个特征的值)："
# )
# X_te = pd.DataFrame(user_in_str.split(","))
# print(X_te)
# # X_te = labelEncoder.transform(X_te)
# print(X_te)
# y_pred = nb.
