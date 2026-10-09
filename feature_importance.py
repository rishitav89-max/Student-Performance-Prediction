import joblib
import pandas as pd

# Load the trained Linear Regression pipeline
model = joblib.load(
    "ml/saved_models/linear_regression.pkl"
)

# Get the preprocessing step
preprocessor = model.named_steps["preprocessor"]

# Get the trained regression model
regressor = model.named_steps["model"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get regression coefficients
coefficients = regressor.coef_

# Create a DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Calculate absolute coefficient
importance_df["Absolute_Coefficient"] = (
    importance_df["Coefficient"].abs()
)

# Sort by absolute coefficient
importance_df = importance_df.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

# Round values
importance_df["Coefficient"] = (
    importance_df["Coefficient"].round(3)
)

importance_df["Absolute_Coefficient"] = (
    importance_df["Absolute_Coefficient"].round(3)
)

print("\nFeature Importance / Model Interpretation")
print("=" * 70)
print(
    importance_df[
        ["Feature", "Coefficient", "Absolute_Coefficient"]
    ].to_string(index=False)
)
print("=" * 70)