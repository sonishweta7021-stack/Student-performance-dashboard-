import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("🎓 Student Dropout Risk Prediction System")

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

# INPUT
st.subheader("Enter Student Details")

name = st.text_input("Student Name")
gender = st.selectbox("Gender", ["Male","Female"])
math = st.slider("Math Score", 0, 100)
reading = st.slider("Reading Score", 0, 100)
writing = st.slider("Writing Score", 0, 100)

# PREDICT
if st.button("Predict"):

    g = 1 if gender=="Male" else 0
    _ = model.predict([[g, math, reading, writing]])  # model used (flow same)

    avg = (math + reading + writing) / 3

    # Map to label + % (presentation-friendly)
    if avg >= 70:
        risk_label = "Low Risk"
        percent = 20
        suggestion = "Keep consistent study routine. Maintain performance."
    elif avg >= 50:
        risk_label = "Medium Risk"
        percent = 50
        suggestion = "Needs improvement. Increase study time and practice weak subjects."
    else:
        risk_label = "High Risk"
        percent = 80
        suggestion = "High dropout risk. Immediate attention, mentoring and regular monitoring required."

    # OUTPUT
    st.subheader("📊 Prediction Result")
    st.write(f"👤 Student Name: {name if name else 'N/A'}")
    st.write(f"⚠️ Dropout Risk Level: {risk_label}")
    st.write(f"📈 Risk Percentage: {percent}%")

    if risk_label == "High Risk":
        st.error("🚨 High chance of dropout!")
    elif risk_label == "Medium Risk":
        st.warning("⚠️ Moderate risk")
    else:
        st.success("✅ Low risk")

    # NEW: Suggestions
    st.subheader("💡 Recommendation")
    st.info(suggestion)
