import joblib
import pandas as pd

# Load the best trained model
model = joblib.load(
    "ml/saved_models/best_model.pkl"
)

# Sample student data
student_data = pd.DataFrame([{
    "age": 16,
    "gender": "Female",
    "attendance_percentage": 90,
    "assignment_score": 85,
    "internal_marks": 88,
    "previous_percentage": 82,
    "study_hours_per_day": 7,
    "class_participation": "High",
    "extracurricular_score": 75
}])

# Predict final percentage
predicted_percentage = model.predict(student_data)[0]

# Keep prediction between 0 and 100
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

print("\nStudent Performance Prediction")
print("=" * 40)

print(
    "Predicted Final Percentage:",
    round(predicted_percentage, 2),
    "%"
)

print(
    "Performance Category:",
    performance_category
)

print("=" * 40)