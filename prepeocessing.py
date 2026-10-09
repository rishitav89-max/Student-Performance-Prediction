import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

# 5. Create Preprocessing Pipeline
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

# 6. Fit Preprocessor on Training Data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 7. Display Information
print("Dataset loaded successfully.")
print("Total records:", len(df))

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nOriginal training features:", X_train.shape[1])
print("Processed training features:", X_train_processed.shape[1])

print("\nPreprocessing completed successfully.")