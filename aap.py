import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Risk Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Feature engineering
df["average_score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3

def risk(score):
    if score >= 70:
        return "Low Risk"
    elif score >= 50:
        return "Medium Risk"
    else:
        return "High Risk"

df["risk"] = df["average_score"].apply(risk)

# Encode
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
df["risk"] = le.fit_transform(df["risk"])

X = df[["gender","math score","reading score","writing score"]]
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 🎯 NAVIGATION
menu = st.sidebar.selectbox("Navigation", ["Dashboard", "Prediction"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("📊 Student Performance Dashboard")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Average Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["average_score"])
    st.pyplot(fig)

    st.subheader("Risk Distribution")
    risk_counts = df["risk"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.bar(["Low","Medium","High"], risk_counts)
    st.pyplot(fig2)

# ---------------- PREDICTION ----------------
if menu == "Prediction":
    st.title("🎓 Student Dropout Risk Prediction")

    name = st.text_input("Student Name")
    gender = st.selectbox("Gender", ["Male","Female"])
    math = st.slider("Math Score", 0, 100)
    reading = st.slider("Reading Score", 0, 100)
    writing = st.slider("Writing Score", 0, 100)

    if st.button("Predict"):

        g = 1 if gender=="Male" else 0
        _ = model.predict([[g, math, reading, writing]])

        avg = (math + reading + writing) / 3

        if avg >= 70:
            risk_label = "Low Risk"
            percent = 20
            suggestion = "Keep consistent study routine."
        elif avg >= 50:
            risk_label = "Medium Risk"
            percent = 50
            suggestion = "Needs improvement and regular practice."
        else:
            risk_label = "High Risk"
            percent = 80
            suggestion = "High dropout risk. Immediate attention required."

        st.subheader("📊 Result")

        st.write(f"👤 Name: {name}")
        st.write(f"⚠️ Risk Level: {risk_label}")
        st.write(f"📈 Risk Percentage: {percent}%")

        if risk_label == "High Risk":
            st.error("🚨 High Risk")
        elif risk_label == "Medium Risk":
            st.warning("⚠️ Medium Risk")
        else:
            st.success("✅ Low Risk")

        st.subheader("💡 Suggestion")
        st.info(suggestion)
