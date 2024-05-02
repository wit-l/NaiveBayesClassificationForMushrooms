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


def process_input(user_input: str, columns: list[str]) -> str:
    try:
        labelEncoders, nb = load_encoders_model("encoders.txt", "model.txt")
    except FileNotFoundError:
        train_model()
        labelEncoders, nb = load_encoders_model("encoders.txt", "model.txt")
    except Exception as e:
        return "发生预期外的错误, 原因：" + str(e.__cause__)
    X = pd.DataFrame([user_input.split(",")], columns=columns)
    i = 1
    for col in X.columns:
        X[col] = labelEncoders[i].transform(X[col])
        i += 1
    y_pred = nb.predict(X)[0]
    result = ""
    if y_pred == 1:
        result = "有毒"
    elif y_pred == 0:
        result = "无毒"
    return "预测结果：" + result


@app.route("/", methods=["GET", "POST"])
def index():
    # read column names
    file = "mushrooms.csv"
    columns = []

    try:
        with open(file, "r") as f:
            columns = f.readline().strip().split(",")[1:]
    except FileNotFoundError:
        return "Can not open " + file
    except Exception as e:
        return "发生预期外的错误, 原因：" + str(e.__cause__)
    if request.method == "POST":
        user_input = request.form.get("user_input")
        result = process_input(str(user_input), columns)
        return result
    return render_template("form.html", columns=columns)


if __name__ == "__main__":
    app.run()
