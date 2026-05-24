from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib

# ============================================
# CREATE FASTAPI APP
# ============================================

app = FastAPI()

# ============================================
# LOAD ARTIFACTS GLOBALLY
# ============================================

model = joblib.load("model.pkl")

columns = joblib.load("columns.pkl")

reference_df = joblib.load("reference_df.pkl")

print("Model and artifacts loaded successfully")

# ============================================
# REMOVE IITs FOR NOW
# ============================================

reference_df = reference_df[

    reference_df["College_Type"] != "IIT"
]

# ============================================
# INPUT SCHEMA
# ============================================

class UserInput(BaseModel):

    rank: int

    category: str

    gender: str

    quota: str

# ============================================
# HOME ROUTE
# ============================================

@app.get("/")
def home():

    return {

        "message": "JOSAA Predictor Backend Running"
    }

# ============================================
# PREDICTION ROUTE
# ============================================

@app.post("/predict")
def predict(user: UserInput):

    # ========================================
    # COPY REFERENCE DATA
    # ========================================

    temp_df = reference_df.copy()

    # ========================================
    # ADD USER INPUT
    # ========================================

    temp_df["Opening Rank"] = user.rank

    temp_df["Quota"] = user.quota

    temp_df["Gender"] = user.gender

    temp_df["Seat Type"] = user.category

    # ========================================
    # FEATURE ENGINEERING
    # ========================================

    temp_df["inst_avg_rank"] = 0

    temp_df["prog_avg_rank"] = 0

    temp_df["inst_prog_avg"] = 0

    # ========================================
    # ONE HOT ENCODING
    # ========================================

    temp_df_encoded = pd.get_dummies(temp_df)

    # ========================================
    # MATCH TRAINING COLUMNS
    # ========================================

    for col in columns:

        if col not in temp_df_encoded.columns:

            temp_df_encoded[col] = 0

    # ========================================
    # REMOVE EXTRA COLUMNS
    # ========================================

    temp_df_encoded = temp_df_encoded[columns]

    # ========================================
    # MODEL PREDICTIONS
    # ========================================

    predictions = model.predict(temp_df_encoded)

    # ========================================
    # STORE PREDICTIONS
    # ========================================

    temp_df["Predicted Closing Rank"] = predictions

    # ========================================
    # CONVERT FLOAT TO INTEGER
    # ========================================

    temp_df["Predicted Closing Rank"] = temp_df[
        "Predicted Closing Rank"
    ].astype(int)

    # ========================================
    # CHANCE LOGIC
    # ========================================

    chances = []

    for pred in predictions:

        # SAFE COLLEGES

        if user.rank <= pred * 0.85:

            chances.append("High")

        # REACHABLE COLLEGES

        elif user.rank <= pred:

            chances.append("Medium")

        # DREAM COLLEGES

        else:

            chances.append("Low")

    temp_df["Chance"] = chances

    # ========================================
    # REMOVE DUPLICATES
    # ========================================

    temp_df = temp_df.drop_duplicates(

        subset=["Institute", "Program"]
    )

    # ========================================
    # HIGH CHANCE COLLEGES
    # ========================================

    high_df = temp_df[

        temp_df["Chance"] == "High"
    ]

    high_df = high_df.sort_values(

        by="Predicted Closing Rank"
    )

    high_df = high_df.head(5)

    # ========================================
    # MEDIUM CHANCE COLLEGES
    # ========================================

    medium_df = temp_df[

        temp_df["Chance"] == "Medium"
    ]

    medium_df = medium_df.sort_values(

        by="Predicted Closing Rank"
    )

    medium_df = medium_df.head(3)

    # ========================================
    # LOW CHANCE COLLEGES
    # ========================================

    low_df = temp_df[

        temp_df["Chance"] == "Low"
    ]

    # CLOSEST DREAM COLLEGES

    low_df["distance"] = abs(

        low_df["Predicted Closing Rank"] - user.rank
    )

    low_df = low_df.sort_values(

        by="distance"
    )

    low_df = low_df.head(2)

    # ========================================
    # FINAL RECOMMENDATIONS
    # ========================================

    final_df = pd.concat([

        high_df,

        medium_df,

        low_df
    ])

    # ========================================
    # FINAL OUTPUT
    # ========================================

    result = final_df[[

        "Institute",

        "Program",

        "Predicted Closing Rank",

        "Chance"
    ]]

    # ========================================
    # RETURN JSON RESPONSE
    # ========================================

    return result.to_dict(

        orient="records"
    )