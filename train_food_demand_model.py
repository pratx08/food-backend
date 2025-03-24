import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

data = pd.read_csv("Cafeteria_Combined_Dataset__Food___Raw_Material___Nationality_.csv")

data['Waste_Percentage'] = data['Wasted_Quantity'] / data['Prepared_Quantity']

X = pd.get_dummies(data[['Day', 'Meal_Time', 'Food_Item', 'Nationality', 'Waste_Percentage']])
y = data['Consumed_Quantity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Model trained locally with MSE: {mse:.2f}")

joblib.dump(model, "food_demand_forecast_model.pkl")
print("✅ Model saved as food_demand_forecast_model.pkl")
