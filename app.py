import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("titanic.csv")  # Make sure the Titanic dataset CSV is in the same folder

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Visualisation", "Prediction", "Model Performance"])

# --------------------- DATA EXPLORATION ---------------------
if page == "Data Exploration":
    st.title("📊 Data Exploration")
    st.write("Overview of the Titanic dataset.")

    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write("Columns & Data Types:")
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.write(df.head())

    st.subheader("Interactive Filtering")
    pclass_filter = st.multiselect("Select Passenger Class", options=df["Pclass"].unique(), default=df["Pclass"].unique())
    filtered_data = df[df["Pclass"].isin(pclass_filter)]
    st.write(filtered_data)

# --------------------- VISUALISATION ---------------------
elif page == "Visualisation":
    st.title("📈 Visualisation")
    st.write("Interactive charts from the Titanic dataset.")

    # Chart 1: Survival count
    st.subheader("Survival Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Survived", ax=ax1)
    st.pyplot(fig1)

    # Chart 2: Age distribution by survival
    st.subheader("Age Distribution by Survival")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x="Age", hue="Survived", kde=True, ax=ax2)
    st.pyplot(fig2)

    # Chart 3: Fare vs Class
    st.subheader("Fare by Class")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x="Pclass", y="Fare", ax=ax3)
    st.pyplot(fig3)

# --------------------- PREDICTION ---------------------
elif page == "Prediction":
    st.title("🚢 Titanic Survival Predictor")
    st.markdown("Enter passenger details to predict survival.")

    st.sidebar.header("Passenger Features")

    def user_input_features():
        Pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
        Sex = st.sidebar.selectbox("Sex", ['male', 'female'])
        Age = st.sidebar.slider("Age", 0, 80, 25)
        SibSp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
        Parch = st.sidebar.number_input("Number of Parents/Children Aboard", 0, 6, 0)
        Fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 50.0)
        Embarked = st.sidebar.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
        Title = st.sidebar.selectbox("Title", ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'])

        # Encoding
        Sex = 1 if Sex == 'male' else 0
        Embarked = {'C': 0, 'Q': 1, 'S': 2}[Embarked]
        Title = {'Mr': 3, 'Miss': 1, 'Mrs': 2, 'Master': 0, 'Rare': 4}[Title]

        data = {
            'Pclass': Pclass,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare,
            'Embarked': Embarked,
            'Title': Title
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    st.subheader("Passenger Data")
    st.write(input_df)

    # Prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display result
    st.subheader("Prediction")
    st.write("🎯 Survived" if prediction[0] == 1 else "💀 Did Not Survive")

    st.subheader("Prediction Probability")
    st.write(f"Survival Probability: {prediction_proba[0][1]:.2f}")

# --------------------- MODEL PERFORMANCE ---------------------
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.write("Evaluation of the trained Titanic model.")

    # Prepare data (assuming features are already preprocessed)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    y_pred = model.predict(X)

    st.subheader("Accuracy Score")
    st.write(accuracy_score(y, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
