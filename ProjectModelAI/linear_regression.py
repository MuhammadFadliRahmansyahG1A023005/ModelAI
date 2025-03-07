import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset: Jam belajar vs Nilai ujian
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

# Membagi dataset menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi dengan data testing
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Visualisasi hasil regresi
plt.scatter(X, y, color='blue', label='Data Aktual')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regresi Linear')
plt.xlabel('Jam Belajar')
plt.ylabel('Nilai Ujian')
plt.legend()
plt.show()
