import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

# Prepare a dictionary to store predictions
predictions = {category: {'Linear': None, 'Polynomial': None, 'Exponential': None, 'RandomForest': None}
               for category in ['C1', 'C2', 'C3', 'G1', 'G2', 'G3']}

combined_data = pd.concat([data_26, data_27])
predict_days = np.array([2, 1, 0.5, 0.125, 0]).reshape(-1, 1)

for category in predictions.keys():
    # Data preparation
    X = combined_data[['Days']]
    y = combined_data[category]
    
    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    predictions[category]['Linear'] = lin_reg.predict(predict_days.reshape(-1, 1))
    
    # Polynomial Regression
    poly_features = PolynomialFeatures(degree=3)
    X_poly = poly_features.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    predict_days_poly = poly_features.transform(predict_days.reshape(-1, 1))
    predictions[category]['Polynomial'] = poly_reg.predict(predict_days_poly)
    
    # Exponential Regression
    try:
        # Example of initial guesses: [1, 0.1, 1]
        # Adjust these based on your understanding of the data
        initial_guesses = [1, 0.1, 1]
        popt, _ = curve_fit(exponential_model, X['Days'].values.ravel(), y, p0=initial_guesses, maxfev=5000)
        predictions[category]['Exponential'] = exponential_model(predict_days.ravel(), *popt)
    except RuntimeError as e:
        print(f"Error fitting Exponential Regression for {category}: {e}")
        predictions[category]['Exponential'] = ['Fit Error'] * len(predict_days.ravel())
    
    # Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100)
    rf_reg.fit(X, y)
    predictions[category]['RandomForest'] = rf_reg.predict(predict_days.reshape(-1, 1))

# Assuming data_28 is defined as follows based on your description
data_28 = pd.DataFrame({
    'Days': [10.125, 9.208, 8.166, 7.125, 6.208, 5.125, 4.125, 3.416, 2, 1, 0.5, 0.125, 0],
    'C1': [1307, 1309, 1310, 1312, 1312, 1312, 1317, 1317, np.nan, np.nan, np.nan, np.nan, np.nan],
    'C2': [1444, 1446, 1449, 1450, 1453, 1456, 1458, 1461, np.nan, np.nan, np.nan, np.nan, np.nan],
    'C3': [1581, 1583, 1586, 1588, 1590, 1593, 1597, 1600, np.nan, np.nan, np.nan, np.nan, np.nan],
    'G1': [1801, 1804, 1808, 1812, 1815, 1818, 1821, 1822, np.nan, np.nan, np.nan, np.nan, np.nan],
    'G2': [1889, 1892, 1894, 1896, 1897, 1900, 1901, 1903, np.nan, np.nan, np.nan, np.nan, np.nan],
    'G3': [1979, 1980, 1984, 1986, 1986, 1989, 1993, 1995, np.nan, np.nan, np.nan, np.nan, np.nan]
})

# Adjusting the plotting to include data points of data_28 up until the prediction days to all graphs
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs = axs.flatten()
colors = ['blue', 'green', 'orange', 'purple']  # Excluding 'orange' for simplicity
model_names = ['Linear', 'Polynomial', 'Exponential', 'RandomForest']  # Excluding 'Exponential' for clarity

for idx, category in enumerate(['C1', 'C2', 'C3', 'G1', 'G2', 'G3']):
    for i, model_name in enumerate(model_names):
        axs[idx].plot(predict_days.ravel(), predictions[category][model_name], marker='o', linestyle='-', color=colors[i], label=model_name)
    axs[idx].scatter(combined_data['Days'], combined_data[category], color='black', alpha=0.5)  # Actual data points from seasons 26 and 27
    axs[idx].scatter(data_28['Days'], data_28[category], color='gray', alpha=0.5)  # Data points from season 28
    axs[idx].axvline(x=0, color='red', linestyle='--')  # Day 0 prediction line
    if category == 'C3':
        axs[idx].axhline(y=1630, color='red', linestyle='-')  # Horizontal line at y=1630 for C3
    axs[idx].set_title(category)
    axs[idx].invert_xaxis()  # Invert x-axis to match the countdown nature of days
    axs[idx].legend()

plt.tight_layout()
plt.show()
