import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Dashboard", layout="wide")

st.sidebar.title("📊 Navigation")
option = st.sidebar.radio("Go to", ["Dashboard", "Prediction"])

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Feature Engineering
df["average_score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3

def risk(score):
    if score >= 70:
        return "Low Risk"
    elif score >= 50:
        return "Medium Risk"
    else:
        return "High Risk"

df["risk"] = df["average_score"].apply(risk)

# Encoding
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
df["lunch"] = le.fit_transform(df["lunch"])
df["test preparation course"] = le.fit_transform(df["test preparation course"])
df["risk"] = le.fit_transform(df["risk"])

# Model
X = df[["gender","math score","reading score","writing score"]]
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ================= DASHBOARD =================
if option == "Dashboard":
    st.title("📊 Student Performance Dashboard")

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Scores Distribution")
    st.bar_chart(df[["math score", "reading score", "writing score"]])

    st.subheader("📊 Average Score Distribution")
    st.line_chart(df["average_score"])

    st.subheader("📌 Basic Statistics")
    st.write(df.describe())

# ================= PREDICTION =================
elif option == "Prediction":
    st.title("🎓 Risk Prediction")

    gender = st.selectbox("Gender", ["Male","Female"])
    math = st.slider("Math Score", 0, 100)
    reading = st.slider("Reading Score", 0, 100)
    writing = st.slider("Writing Score", 0, 100)

    if st.button("Predict"):
        g = 1 if gender=="Male" else 0
        result = model.predict([[g, math, reading, writing]])

        if result == 0:
            st.error("High Risk")
        elif result == 1:
            st.warning("Medium Risk")
        else:
            st.success("Low Risk")
