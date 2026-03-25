import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


data = pd.read_csv("train.csv")

print("Dataset Loaded Successfully")
print(data.head())


data = data.dropna()

data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

data['year'] = data['pickup_datetime'].dt.year
data['month'] = data['pickup_datetime'].dt.month
data['day'] = data['pickup_datetime'].dt.day
data['hour'] = data['pickup_datetime'].dt.hour

data = data.drop(['pickup_datetime'], axis=1)


X = data.drop("fare_amount", axis=1)
y = data["fare_amount"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

print("Model Training Completed")


predictions = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Model RMSE:", rmse)


results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print(results.head())


plt.scatter(y_test, predictions)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Taxi Fare Prediction")
plt.show()