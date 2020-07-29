import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np


def predict(arguments):
    try:
        data = pd.DataFrame(arguments, index=[0])

        for col_name in data:
            data[col_name] = pd.to_numeric(data[col_name], errors="ignore")

        inverse_columns = [col_name for col_name in data if "inv_" in col_name]

        def invert(value):
            if type(value) is str:
                return 0
            else:
                return 1 / 1 if value == 0 else 1 / value

        for col_name in inverse_columns:
            data[col_name] = data[col_name].map(invert)

        for joint_col, indiv_col in zip(
            ["annual_inc_joint", "dti_joint"], ["annual_inc", "dti"]
        ):
            data[joint_col] = [
                indiv_val if type(joint_val) is str else joint_val
                for joint_val, indiv_val in zip(data[joint_col], data[indiv_col])
            ]

        transformer = joblib.load("./models/data_transformer.joblib")
        X = transformer.transform(data)

        model = load_model("./models/loan_risk_model")
        return model(X).numpy()[0][0]

    except:
        return np.nan
