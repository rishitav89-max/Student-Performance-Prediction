import requests

url = "http://127.0.0.1:8000/api/predict/"

student_data = {
    "student_id": "STU_TEST_002",
    "age": 16,
    "gender": "Female",
    "attendance_percentage": 120,
    "assignment_score": 85,
    "internal_marks": 88,
    "previous_percentage": 82,
    "study_hours_per_day": 7,
    "class_participation": "High",
    "extracurricular_score": 75
}

response = requests.post(
    url,
    json=student_data
)

print("Status Code:", response.status_code)
print("Response:", response.json())