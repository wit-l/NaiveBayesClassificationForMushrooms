#!/opt/anaconda3/bin/python3
import pandas as pd
import pickle
from train import train_model
from typing import List, Tuple
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template, redirect


def load_encoders_model(
    label_encoders_file: str, model_file: str
) -> Tuple[List[LabelEncoder], GaussianNB]:
    with open(label_encoders_file, "rb") as les_, open(model_file, "rb") as model_:
        les = pickle.load(les_)
        model = pickle.load(model_)
    return les, model


app = Flask(__name__)
accuracy_score = 0.0
precision_score = 0.0
recall_score = 0.0


def process_input(user_input: str, columns: list[str]) -> str:
    global accuracy_score, precision_score, recall_score
    try:
        labelEncoders, nb = load_encoders_model("encoders.txt", "model.txt")
    except FileNotFoundError:
        accuracy_score, precision_score, recall_score = train_model()
        labelEncoders, nb = load_encoders_model("encoders.txt", "model.txt")
    except Exception as e:
        return "发生预期外的错误, 原因：" + str(e.__cause__)
    X = pd.DataFrame([user_input.split(",")], columns=columns)
    i = 1
    for col in X.columns:
        X[col] = labelEncoders[i].transform(X[col])
        i += 1
    y_pred = nb.predict(X)[0]
    return y_pred


@app.route("/", methods=["GET", "POST"])
def index():
    # read column names
    file = "mushrooms.csv"
    columns = []
    result = ""

    try:
        with open(file, "r") as f:
            columns = f.readline().strip().split(",")[1:]
    except FileNotFoundError:
        return "Can not open " + file
    except Exception as e:
        return "发生预期外的错误, 原因：" + str(e.__cause__)

    if request.method == "POST":
        user_input = request.form["user_input"]
        if user_input == "":
            return redirect("/")
        result = process_input(str(user_input), columns)

    return render_template(
        "form.html",
        columns=columns,
        accuracy_score=accuracy_score,
        precision_score=precision_score,
        recall_score=recall_score,
        result=result,
    )


@app.route("/train", methods=["POST"])
def invoke_train():
    global accuracy_score, precision_score, recall_score
    accuracy_score, precision_score, recall_score = train_model()
    return redirect("/")


if __name__ == "__main__":
    app.run()
