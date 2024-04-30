import pandas as pd
import pickle
from train import train_model

if __name__ == "__main__":
    # read column names
    file = "mushrooms.csv"
    columns = []
    # TODO: The exception should be handled on the outside function
    try:
        with open(file, "r") as f:
            columns = f.readline().strip().split(",")[1:]
    except FileNotFoundError:
        print("Can not open %s." % (file))
        exit(-1)
    except Exception as e:
        print("发生预期外的错误, 原因：", e.__cause__)
        exit(-1)

    # get encoders from file
    file = "encoders.txt"
    try:
        with open(file, "rb") as f:
            labelEncoders = pickle.load(f)
    except FileNotFoundError:
        # print("Can not open %s. Training the model first." % (file))
        train_model()
        with open(file, "rb") as f:
            labelEncoders = pickle.load(f)
    except Exception as e:
        print("发生预期外的错误, 原因：", e.__cause__)
        exit(-1)

    # read file to get model object
    file = "model.txt"
    try:
        with open(file, "rb") as f:
            nb = pickle.load(f)
    except FileNotFoundError:
        # print("Can not open %s. Training the model first." % (file))
        train_model()
        with open(file, "rb") as f:
            nb = pickle.load(f)
    except Exception as e:
        print("发生预期外的错误, 原因：", e.__cause__)
        exit(-1)

    for i in columns:
        print(i, end=" ")
    print("\n")
    user_in_str = input(
        "请输入蘑菇的各个特征值(按照以上特征的顺序输入，并用','分隔各个特征的值)："
    )

    X_te = pd.DataFrame([user_in_str.split(",")], columns=columns)
    # print(X_te)

    i = 1
    for col in X_te.columns:
        X_te[col] = labelEncoders[i].transform(X_te[col])
        i += 1

    y_pred = nb.predict(X_te)[0]
    if y_pred == 1:
        print("有毒")
    elif y_pred == 0:
        print("无毒")
    else:
        print("error")
