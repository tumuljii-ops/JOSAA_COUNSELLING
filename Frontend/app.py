import streamlit as st
import requests
import pandas as pd

# ============================================
# PAGE TITLE
# ============================================

st.title("JOSAA College Predictor")

st.write(

    "Enter your details to get top college recommendations"
)

# ============================================
# USER INPUTS
# ============================================

rank = st.number_input(

    "Enter JEE Rank",

    min_value=1
)

category = st.selectbox(

    "Select Category",

    [

        "OPEN",

        "OBC-NCL",

        "SC",

        "ST",

        "EWS"
    ]
)

gender = st.selectbox(

    "Select Gender",

    [

        "Gender-Neutral",

        "Female-only"
    ]
)

quota = st.selectbox(

    "Select Quota",

    [

        "HS",

        "OS",

        "AI"
    ]
)

# ============================================
# PREDICT BUTTON
# ============================================

if st.button("Predict Colleges"):

    # ========================================
    # CREATE JSON PAYLOAD
    # ========================================

    user_data = {

        "rank": rank,

        "category": category,

        "gender": gender,

        "quota": quota
    }

    # ========================================
    # SEND POST REQUEST
    # ========================================

    response = requests.post(

        "http://josaa-backend:8000/predict",

        json=user_data
    )

    # ========================================
    # RESPONSE SUCCESS
    # ========================================

    if response.status_code == 200:

        predictions = response.json()

        df = pd.DataFrame(predictions)

        st.subheader(

            "Top College Recommendations"
        )

        st.dataframe(df)

    # ========================================
    # ERROR
    # ========================================

    else:

        st.error(

            "Backend Error"
        )