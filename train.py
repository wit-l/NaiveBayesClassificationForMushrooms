import pandas as pd

# import numpy as np

import pickle

# from IPython.display import display

# LabelEncoder 用于标签编码 例['red', 'green', 'blue'] => [0, 1, 2]
from sklearn.preprocessing import LabelEncoder

# train_test_split 分割训练集和测试集
from sklearn.model_selection import (
    train_test_split,
    # GridSearchCV
)

# 高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    # classification_report,
    confusion_matrix,
    # accuracy_score,
    # precision_score,
    # recall_score,
)


def train_model() -> list[float]:
    file = "mushrooms.csv"
    data = pd.read_csv(file)

    # print("读入csv后，存放samples的data中前5行内容：")
    # display(data.head())

    # print("类别：\n", data["class"].unique())

    # print("各类别的数量")
    # print(data['class'].value_counts())
    # print(data.groupby("class").size())

    # 编码方案一：
    # labelEncoder = LabelEncoder()
    # data = data.apply(labelEncoder.fit_transform)

    # print(type(data.columns))
    # print(data.columns)

    # 编码方案二:
    # 保存编码方案，后续输入特征时需要使用同一套编码方案
    label_encoders = [LabelEncoder() for _ in range(data.shape[1])]
    i = 0
    # print(type(data.columns))
    for col in data.columns:
        data[col] = label_encoders[i].fit_transform(data[col])
        i += 1
    with open("encoders.txt", "wb") as f:
        pickle.dump(label_encoders, f)

    # print("对属性值和类别编码后得到的数据：")
    # display(data.head())

    # samples
    X = data.iloc[:, 1:]  # DataFrame
    # 也可通过drop去掉第一列：
    # X = data.drop("class", axis=1)

    # Labels
    y = data.iloc[:, 0]  # Series
    # 也可通过索引取出第一列：
    # y = data["class"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    # nb = GaussianNB()
    nb = GaussianNB(var_smoothing=2.310129700083158e-4)

    # 超参数调优
    # param_grid = {"var_smoothing": np.logspace(0, -9, num=100)}
    # grid = GridSearchCV(nb, param_grid, cv=5)
    # grid.fit(X_train, y_train)
    # print("最优参数：", grid.best_params_)
    # print("最佳交叉验证得分：{:.2f}".format(grid.best_score_))

    # training
    nb.fit(X_train, y_train)

    # store model object in the file
    with open("model.txt", "wb") as f:
        pickle.dump(nb, f)

    y_predict = nb.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)

    # print("前四十五个测试数据的真实类别编号：")
    # print(np.array(y_test)[:45])
    #
    # print("\n前四十五个测试数据的预测类别编号：")
    # print(y_predict[:45])

    # print("confusion matrix:\n", cm)
    # print("混淆矩阵类型:", type(cm))

    # [[TN, FP],
    #  [FN, TP]]
    tp = cm[1, 1]  # 真阳
    fp = cm[0, 1]  # 假阳
    fn = cm[1, 0]  # 假阴
    tn = cm[0, 0]  # 真阴

    sum = tp + fp + fn + tn
    accuracy_rate = (tp + tn) / sum
    precision_rate = tp / (tp + fp)
    recall_rate = tp / (tp + fn)

    # accuracy_rate = (tp + tn) / np.sum(cm)
    # precision_rate = tp / np.sum(cm, 1)[0]
    # recall_rate = tp / np.sum(cm, 0)[0]

    # print("准确率", accuracy_rate)
    # print("精确率", precision_rate)
    # print("召回率", recall_rate)

    # print("准确率", accuracy_score(y_test, y_predict))
    # print("精确率", precision_score(y_test, y_predict))
    # print("召回率", recall_score(y_test, y_predict))

    return [accuracy_rate, precision_rate, recall_rate]

    # print(classification_report(y_test, y_predict, target_names=["无毒", "有毒"]))
