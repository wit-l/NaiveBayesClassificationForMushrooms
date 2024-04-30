#!/opt/anaconda3/bin/python3
import pandas as pd
import pickle
from train import train_model
from typing import List, Tuple
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template


def load_encoders_model(
    label_encoders_file: str, model_file: str
) -> Tuple[List[LabelEncoder], GaussianNB]:
    with open(label_encoders_file, "rb") as les_, open(model_file, "rb") as model_:
        les = pickle.load(les_)
        model = pickle.load(model_)
    return les, model


app = Flask(__name__)


def process_input(user_input) -> str:
    return "result of handling:" + user_input


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form.get("user_input")
        result = process_input(user_input)
        return result
    return render_template("form.html", columns=columns)


if __name__ == "__main__":
    app.run()
    # read column names
    file = "mushrooms.csv"
    columns = []
    try:
        with open(file, "r") as f:
            columns = f.readline().strip().split(",")[1:]
    except FileNotFoundError:
        print("Can not open %s." % (file))
        exit(-1)
    except Exception as e:
        print("发生预期外的错误, 原因：", e.__cause__)
        exit(-1)

    try:
        labelEncoders, nb = load_encoders_model("encoders.txt", "model.txt")
    except FileNotFoundError:
        train_model()
        labelEncoders, nb = load_encoders_model("encoders.txt", "model.txt")
    except Exception as e:
        print("发生预期外的错误, 原因：", e.__cause__)
        exit(-1)

    for i in columns:
        print(i, end=" ")
    print("\n")
    user_in_str = input(
        "请输入蘑菇的各个特征值(按照以上特征的顺序输入，并用','分隔各个特征的值)："
    )

    X = pd.DataFrame([user_in_str.split(",")], columns=columns)

    i = 1
    for col in X.columns:
        X[col] = labelEncoders[i].transform(X[col])
        i += 1

    y_pred = nb.predict(X)[0]
    if y_pred == 1:
        print("有毒")
    elif y_pred == 0:
        print("无毒")
    else:
        print("error")
