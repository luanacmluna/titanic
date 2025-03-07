import streamlit as st
import pickle
import numpy as np

# Load the trained models
model_files = {
    "Logistic Regression": "logistic_regression_titanic.pkl",
    "Random Forest": "random_forest_titanic.pkl",
    "Support Vector Machine": "svm_titanic.pkl"
}

models = {}
for name, file in model_files.items():
    try:
        with open(file, 'rb') as f:
            models[name] = pickle.load(f)
    except FileNotFoundError:
        st.warning(f"⚠️ Model file {file} not found. {name} will not be available.")

st.title("Titanic Survival Prediction App")

# Model selection
st.header("Select a Model")
model_choice = st.radio("Choose a model:", list(models.keys()))

# Display model descriptions
descriptions = {
    "Logistic Regression": "A simple model for binary classification using a linear decision boundary.",
    "Random Forest": "An ensemble model combining multiple decision trees for better accuracy.",
    "Support Vector Machine": "A model that finds the optimal hyperplane to classify passengers."
}
st.write(descriptions.get(model_choice, "No description available."))

# User input section
st.header("Enter Passenger Details")

def input_with_range(label, min_val, max_val, step=None, dtype=float):
    if step is None:
        step = 1 if dtype == int else 0.1
    return st.number_input(f"{label} (Min: {min_val}, Max: {max_val})", min_value=min_val, max_value=max_val, step=step, format="%.2f" if dtype == float else "%d")

Pclass = input_with_range("Passenger Class (1 = First, 2 = Second, 3 = Third)", 1, 3, dtype=int)
Sex = st.radio("Sex", ["Male", "Female"])
Age = input_with_range("Age", 0, 100, dtype=int)
SibSp = input_with_range("Number of Siblings/Spouses Aboard", 0, 10, dtype=int)
Parch = input_with_range("Number of Parents/Children Aboard", 0, 10, dtype=int)
Fare = input_with_range("Fare", 0.0, 500.0, step=0.01, dtype=float)

# Encode categorical values
Sex_encoded = 1 if Sex == "Male" else 0

# Prediction button
if st.button("Predict Survival"):
    input_data = np.array([[Pclass, Sex_encoded, Age, SibSp, Parch, Fare]])
    
    if model_choice in models:
        prediction = models[model_choice].predict(input_data)
        result = "Survived" if prediction[0] == 1 else "Did Not Survive"
        st.success(f"Prediction: {result}")
    else:
        st.error("Model not available. Please check if the model files exist.")