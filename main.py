#!/opt/anaconda3/envs/sklearn/bin/python3
from sys import excepthook
import pandas as pd
import pickle
from typing import List, Tuple
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from flask import Flask, redirect, render_template, request, send_from_directory
from train import train_model


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
columns = []
download_result_file = False
result = ""


def process_input(user_input: str, columns: list[str]) -> str:
    global accuracy_score, precision_score, recall_score
    try:
        label_encoders, nb = load_encoders_model("encoders.txt", "model.txt")
    except FileNotFoundError:
        accuracy_score, precision_score, recall_score = train_model()
        label_encoders, nb = load_encoders_model("encoders.txt", "model.txt")
    except Exception as e:
        return "发生预期外的错误, 原因：" + str(e.__doc__) + "！！！"
    X = pd.DataFrame([user_input.split(",")], columns=columns)  # type: ignore
    i = 1
    for col in X.columns:
        X[col] = label_encoders[i].transform(X[col])
        i += 1
    y_pred = nb.predict(X)[0]
    return y_pred


@app.route("/", methods=["GET", "POST"])
def index():
    global result
    result = ""
    global download_result_file
    global columns
    if columns.__len__() == 0:
        # read column names
        file = "mushrooms.csv"
        try:
            with open(file, "r") as f:
                columns = f.readline().strip().split(",")[1:]
            # raise Exception
        except FileNotFoundError:
            result = "Can not open " + file + "!!!"
        except Exception as e:
            result = "发生预期外的错误, 原因：" + str(e.__doc__) + "！！！"

    if request.method == "POST":
        user_input = request.form["sample_text"]
        file = request.files["sample_file"]

        if file.filename != "":
            download_result_file = True
            with open("./result.txt", "w", encoding="utf-8") as result_file:
                for line in file.stream:
                    # print(line.decode().strip())
                    try:
                        result = process_input(line.decode().strip(), columns)
                    except ValueError:
                        result = "输入格式不正确，请严格按照以上顺序输入完整！！！"
                    except Exception as e:
                        result = "发生预期外的错误, 原因：" + str(e.__doc__) + "！！！"
                    if result == 0:
                        result = str("无毒")
                    elif result == 1:
                        result = str("有毒")
                    result_file.write(result)
                    result_file.write("\n")
                    # print(result)
            if user_input == "":
                return render_template(
                    "form.html",
                    columns=columns,
                    accuracy_score=0,
                    result="",
                    download_result_file=download_result_file,
                )

        try:
            result = process_input(str(user_input), columns)
        except ValueError:
            result = "输入格式不正确，请严格按照以上顺序输入完整！！！"
        except Exception as e:
            result = "发生预期外的错误, 原因：" + str(e.__doc__) + "！！！"

    return render_template(
        "form.html",
        columns=columns,
        accuracy_score=accuracy_score,
        precision_score=precision_score,
        recall_score=recall_score,
        result=result,
        download_result_file=download_result_file,
    )


@app.route("/train", methods=["POST"])
def invoke_train():
    try:
        global accuracy_score, precision_score, recall_score, result
        accuracy_score, precision_score, recall_score = train_model()
        return render_template(
            "form.html",
            columns=columns,
            result=result,
            download_result_file=download_result_file,
            accuracy_score=accuracy_score,
            precision_score=precision_score,
            recall_score=recall_score,
        )
    except Exception as e:
        return "发生预期外的错误, 原因：" + str(e.__doc__) + "!!!"


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    return send_from_directory("./", filename, as_attachment=True)


if __name__ == "__main__":
    app.run()
