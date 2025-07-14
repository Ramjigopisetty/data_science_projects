import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\Ram\Downloads\archive (4)\ADVERTISING.csv")
print("Dataset Head:\n", df.head())

print("\nNull values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

X = df[['TV', 'Radio', 'Newspaper']] 
y = df['Sales']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Mean Squared Error: {mse:.2f}")
print(f"📊 R² Score (Model Accuracy): {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test.values, label='Actual Sales', color='blue', linewidth=2)
plt.plot(range(len(y_pred)), y_pred, label='Predicted Sales', color='red', linewidth=2)
plt.title("Actual vs Predicted Sales (Line Plot)")
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def predict_sales(tv, radio, newspaper):
    sample = pd.DataFrame({
        'TV': [tv],
        'Radio': [radio],
        'Newspaper': [newspaper]
    })
    prediction = model.predict(sample)[0]
    return round(prediction, 2)

A_tv, B_radio, C_newspaper= input("Enter No.of TV's, Radios,Newspapers:-").split(" ")
sample_prediction = predict_sales(A_tv, B_radio, C_newspaper)
print("\n📦 Predicted Sales for TV={}, Radio={}, Newspaper={}: {}".format(A_tv, B_radio, C_newspaper,sample_prediction))

#now let's run the program and see the difference between the predicted sales and actual sale with plot graph