import os
import joblib
import pandas as pd

# Get the project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# Path of the trained ML model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "saved_models",
    "best_model.pkl"
)

# Load the trained model
model = joblib.load(MODEL_PATH)

def predict_student_performance(student_data):
    # Convert input data into DataFrame
    input_data = pd.DataFrame([student_data])

    # Make prediction
    predicted_percentage = model.predict(input_data)[0]

    # Keep percentage between 0 and 100
    predicted_percentage = max(
        0,
        min(100, predicted_percentage)
    )

    # Determine performance category
    if predicted_percentage >= 85:
        performance_category = "Excellent"

    elif predicted_percentage >= 70:
        performance_category = "Good"

    elif predicted_percentage >= 50:
        performance_category = "Average"

    else:
        performance_category = "At Risk"

    return {
        "predicted_percentage": round(
            predicted_percentage, 2
        ),
        "performance_category": performance_category
    }