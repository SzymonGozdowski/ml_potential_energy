import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data1 = pd.read_csv("arduino_data31032025.txt")
data2 = pd.read_csv("arduino_data31032025v2_woda.txt")

rows1 = len(data1)
rows2 = len(data2)

X1 = np.array([i for i in range(rows1)])
X2 = np.array([i for i in range(rows2)])

X1 = np.arange(rows1).reshape(-1, 1)
X2 = np.arange(rows2).reshape(-1, 1)

model = LinearRegression()
model.fit(X1, data1)

y_pred = model.predict(X2)

mse = mean_squared_error(data2, y_pred)
print(f"mse: {mse}")

plt.figure(1)
plt.plot(X2, y_pred, label="predicted")
plt.plot(data2, label="actual data")
plt.legend()
plt.show()
