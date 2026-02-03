# ML-Project-1

## AIM:
To write a program that does the following using sklearn (Linear Regression) and matplotlib :

1.Create a scatter plot between cylinder vs Co2Emission (green color)

2.Using scatter plot compares data cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors

3.Using scatter plot compares data cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors

4.Trains the model with independent variable as cylinder and dependent variable as Co2Emission

5.Trains a new model with independent variable as FuelConsumption_comb and dependent variable as Co2Emission

6.Train model on different train test ratio and train the models and note down their accuracies


## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1.Load the Fuel Consumption dataset using Pandas.

2.Visualize the relationship between vehicle features (cylinders, engine size, fuel consumption) and CO₂ emissions using scatter plots.

3.Select Cylinders as the input feature and CO₂ Emissions as the target variable.

4.Split the dataset into training and testing sets using the train-test split method.

5.Train a Linear Regression model using the training data.

6.Predict CO₂ emissions on the test data and evaluate the model using the R² score.

7.Repeat the training and evaluation process using Fuel Consumption (Combined) as the input feature.

8.Analyze model performance by varying different train–test split ratios.

## PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv("FuelConsumption.csv")

#Q1
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.title("Cylinders vs CO2 Emissions")
plt.show()

#Q2
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinders')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size')

plt.xlabel("Engine Parameters")
plt.ylabel("CO2 Emissions")
plt.title("Cylinders & Engine Size vs CO2 Emissions")
plt.legend()
plt.show()

#Q3
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinders')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='red', label='Fuel Consumption (Combined)')

plt.xlabel("Vehicle Parameters")
plt.ylabel("CO2 Emissions")
plt.title("Multiple Features vs CO2 Emissions")
plt.legend()
plt.show()

#Q4
X = df['CYLINDERS'].values.reshape(-1, 1)
y = df['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy (Cylinders):", r2_score(y_test, y_pred))

#Q5
X = df['FUELCONSUMPTION_COMB'].values.reshape(-1, 1)
y = df['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy (Fuel Consumption):", r2_score(y_test, y_pred))

#Q6
ratios = [0.1, 0.2, 0.3, 0.4]

print("Train-Test Ratio Analysis (Fuel Consumption Model):")
for r in ratios:
    X = df[['FUELCONSUMPTION_COMB']]
    y = df['CO2EMISSIONS']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=r, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = r2_score(y_test, y_pred)
    print(f"Test size = {r} → Accuracy = {acc:.4f}")
```

## OUTPUT:
<img width="638" height="473" alt="Screenshot 2026-02-03 140848" src="https://github.com/user-attachments/assets/063549dd-02f8-453c-a9af-f8713605d8dc" />
<img width="632" height="482" alt="Screenshot 2026-02-03 140859" src="https://github.com/user-attachments/assets/504965c0-059b-4707-b49e-53b4d8ee8e62" />
<img width="629" height="478" alt="Screenshot 2026-02-03 140908" src="https://github.com/user-attachments/assets/4d7242a3-ebd0-4723-bcf0-2643ff8ef4e9" />

## RESULT:
The Fuel Consumption dataset was successfully analyzed using data visualization and Linear Regression.


