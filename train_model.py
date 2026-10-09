import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 1. Load Dataset
DATA_PATH = "dataset/student_performance_dataset.csv"
df = pd.read_csv(DATA_PATH)

# 2. Separate Features and Target
X = df.drop(
    columns=[
        "student_id",
        "final_percentage",
        "performance_category"
    ]
)
y = df["final_percentage"]

# 3. Define Feature Types
numeric_features = [
    "age",
    "attendance_percentage",
    "assignment_score",
    "internal_marks",
    "previous_percentage",
    "study_hours_per_day",
    "extracurricular_score"
]

categorical_features = [
    "gender",
    "class_participation"
]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 5. Create Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# 6. Create ML Models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=6
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=8
    )
}

# 7. Train Models
trained_models = {}
for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )
    pipeline.fit(X_train, y_train)
    trained_models[model_name] = pipeline
    print(f"{model_name} trained successfully.")

# Save the best performing model
best_model = models["Linear Regression"]
best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

best_pipeline.fit(X_train, y_train)

joblib.dump(
    best_pipeline,
    "ml/saved_models/best_model.pkl"
)

print("\nBest model saved successfully.")
print("Model: Linear Regression")
print("File: ml/saved_models/best_model.pkl")


# 8. Save Trained Models
import os
os.makedirs("saved_models", exist_ok=True)
for model_name, pipeline in trained_models.items():
    file_name = (
        model_name.lower()
        .replace(" ", "_")
        + ".pkl"
    )
    file_path = os.path.join(
        "ml/saved_models",
        file_name
    )
    joblib.dump(pipeline, file_path)
    print(f"{model_name} saved at: {file_path}")

# 9. Training Completed
print("\nAll models trained and saved successfully.")