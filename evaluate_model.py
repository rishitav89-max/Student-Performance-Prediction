import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

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

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 4. Load Trained Models
models = {
    "Linear Regression": joblib.load(
        "ml/saved_models/linear_regression.pkl"
    ),

    "Decision Tree": joblib.load(
        "ml/saved_models/decision_tree.pkl"
    ),

    "Random Forest": joblib.load(
        "ml/saved_models/random_forest.pkl"
    )
}

# 5. Evaluate Each Model
results = []
for model_name, model in models.items():
    predictions = model.predict(X_test)
    mae = mean_absolute_error(
        y_test,
        predictions
    )
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

# 6. Display Results
results_df = pd.DataFrame(results)
results_df["MAE"] = results_df["MAE"].round(2)
results_df["RMSE"] = results_df["RMSE"].round(2)
results_df["R2 Score"] = results_df["R2 Score"].round(3)
print("\nModel Evaluation Results")
print("=" * 60)
print(results_df.to_string(index=False))

# 7. Select Best Model
best_model_name = results_df.loc[
    results_df["R2 Score"].idxmax(),
    "Model"
]
print("\n" + "=" * 60)
print("Best Model:", best_model_name)
print("=" * 60)