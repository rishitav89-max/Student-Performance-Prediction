import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained Linear Regression pipeline
model = joblib.load(
    "ml/saved_models/linear_regression.pkl"
)

# Get preprocessing and regression steps
preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["model"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get coefficients
coefficients = regressor.coef_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Sort by coefficient
importance_df = importance_df.sort_values(
    by="Coefficient"
)

# Create feature names for better display
importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
)

# Create chart
plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["Feature"],
    importance_df["Coefficient"]
)

plt.axvline(
    x=0,
    linewidth=1
)

plt.xlabel("Regression Coefficient")
plt.ylabel("Features")
plt.title("Feature Influence on Student Performance")
plt.tight_layout()

# Save chart
plt.savefig(
    "ml/feature_importance_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()