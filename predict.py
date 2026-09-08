# Cognifyz Technologies — Machine Learning Internship
# Task 1: Predict Restaurant Ratings

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. Load Dataset
df = pd.read_csv('restaurant_dataset.csv')

print("First 5 rows:")
display(df.head())

print("Shape:", df.shape)

print("\nDataset Information:")
df.info()


# 3. Explore Dataset
print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

print("\nTarget Statistics:")
print(df['Aggregate rating'].describe())


# Distribution of Restaurant Ratings
plt.figure(figsize=(8, 5))
plt.hist(df['Aggregate rating'], bins=20)
plt.xlabel('Aggregate Rating')
plt.ylabel('Number of Restaurants')
plt.title('Distribution of Restaurant Ratings')
plt.show()


# 4. Select Features and Target
features = [
    'City',
    'Cuisines',
    'Average Cost for two',
    'Has Table booking',
    'Has Online delivery',
    'Is delivering now',
    'Price range',
    'Votes'
]

X = df[features].copy()
y = df['Aggregate rating'].copy()


# 5. Preprocessing
categorical_features = [
    'City',
    'Cuisines',
    'Has Table booking',
    'Has Online delivery',
    'Is delivering now'
]

numeric_features = [
    'Average Cost for two',
    'Price range',
    'Votes'
]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# 7. Linear Regression
linear_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_predictions)

print("\nLinear Regression Results")
print("-------------------------")
print("MAE :", round(linear_mae, 4))
print("MSE :", round(linear_mse, 4))
print("RMSE:", round(linear_rmse, 4))
print("R²  :", round(linear_r2, 4))


# 8. Random Forest Regression
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Regression Results")
print("--------------------------------")
print("MAE :", round(rf_mae, 4))
print("MSE :", round(rf_mse, 4))
print("RMSE:", round(rf_rmse, 4))
print("R²  :", round(rf_r2, 4))


# 9. Compare Models
results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Random Forest Regression'
    ],
    'MAE': [
        linear_mae,
        rf_mae
    ],
    'MSE': [
        linear_mse,
        rf_mse
    ],
    'RMSE': [
        linear_rmse,
        rf_rmse
    ],
    'R2': [
        linear_r2,
        rf_r2
    ]
})

print("\nModel Comparison:")
display(results.round(4))


# 10. Actual vs Predicted Ratings
plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.35
)

plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Restaurant Ratings')

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

plt.show()


# 11. Feature Importance
feature_names = rf_model.named_steps[
    'preprocessor'
].get_feature_names_out()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.named_steps[
        'model'
    ].feature_importances_
})

importance_df = importance_df.sort_values(
    'Importance',
    ascending=False
)

print("\nTop 15 Influential Features:")
display(importance_df.head(15))


# Top 10 Feature Importance Visualization
top_features = importance_df.head(10).sort_values(
    'Importance'
)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features['Feature'],
    top_features['Importance']
)

plt.xlabel('Importance')
plt.title('Top 10 Influential Features')
plt.tight_layout()
plt.show()


# 12. Example Prediction
sample_restaurant = X.iloc[[0]]

predicted_rating = rf_model.predict(
    sample_restaurant
)[0]

print("\nExample Prediction")
print("------------------")
print("Predicted rating:", round(predicted_rating, 2))
print("Actual rating:", y.iloc[0])


# 13. Final Conclusion
print("\nConclusion")
print("----------")

if rf_r2 > linear_r2:
    print("Random Forest performed better than Linear Regression.")
else:
    print("Linear Regression performed better than Random Forest.")

print(
    f"Random Forest R² Score: {rf_r2:.4f}"
)

print(
    f"Random Forest RMSE: {rf_rmse:.4f}"
)

print(
    "Feature importance was analyzed to identify the most influential "
    "features for restaurant rating prediction."
)