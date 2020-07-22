from flask import Flask, Request, request, render_template
from werkzeug.datastructures import ImmutableOrderedMultiDict
from predict import predict
import pandas as pd


class OrderedRequest(Request):
    parameter_storage_class = ImmutableOrderedMultiDict


class MyFlask(Flask):
    request_class = OrderedRequest


app = MyFlask(__name__, static_folder="public", template_folder="views")


@app.route("/")
def homepage():
    return render_template("index.html")


@app.route("/api/predict")
def loan_risk_predict():
    assert isinstance(request.args, ImmutableOrderedMultiDict)

    prediction = predict(request.args)

    if pd.isna(prediction):
        return {"error": "There's something wrong with your input."}, 400

    if prediction < 0:
        prediction = 0
    elif prediction > 1:
        prediction = 1

    description = f"This loan is predicted to recover {round(prediction * 100, 1)}% of its expected return."

    return {"value": str(prediction), "description": description}


if __name__ == "__main__":
    app.run()
