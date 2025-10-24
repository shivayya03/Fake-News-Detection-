import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# -------------------------
# Title
# -------------------------
st.title("📰 Fake News Detection App")
st.write("Enter a news article or sentence, and the model will predict whether it's real or fake.")

# -------------------------
# Load or train model
# -------------------------

MODEL_PATH = "fmodel.pkl"
VECTORIZER_PATH = "tfidf.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
else:
    st.info("Training model for the first time. Please wait...")
    # Load dataset
    df = pd.read_csv("fakenews.csv")  # Replace with your dataset
    df["target"] = LabelEncoder().fit_transform(df["target"])  # Convert 'fake'/'real' to 0/1

    X = df["Text"].str.lower()
    y = df["target"]

    tfidf = TfidfVectorizer(stop_words='english', max_features=9000)
    X_vect = tfidf.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vect, y)

    # Save for future use
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)

# -------------------------
# User Input
# -------------------------

news_input = st.text_area("Enter News Text", height=200)

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        vect_input = tfidf.transform([news_input.lower()])
        prediction = model.predict(vect_input)[0]
        probability = model.predict_proba(vect_input)[0][prediction]

        if prediction == 1:
            st.success(f"✅ This news appears to be **REAL**. (Confidence: {probability:.2%})")
        else:
            st.error(f"🚨 This news appears to be **FAKE**. (Confidence: {probability:.2%})")
