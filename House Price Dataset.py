import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample dataset: area (sq ft) -> price (lakhs)
X = np.array([
    [500], [600], [700], [800], [900],
    [1000], [1100], [1200], [1300], [1400],
    [1500], [1600], [1700], [1800], [2000]
])

y = np.array([
    20, 24, 28, 32, 36,
    40, 44, 48, 52, 56,
    60, 64, 68, 72, 80
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Actual Prices:", y_test)
print("Predicted Prices:", y_pred)
print("\nCoefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# Predict new house price
area = float(input("\nEnter house area in square feet: "))
predicted_price = model.predict([[area]])
print("Predicted House Price: ₹", round(predicted_price[0], 2), "Lakhs")

# Plot regression line
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price (Lakhs)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.show()