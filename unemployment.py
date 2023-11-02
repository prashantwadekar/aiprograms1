import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA

# Load the dataset
data = pd.read_csv('unemployment.csv')  # Replace 'unemployment.csv' with the actual file name

# Convert the date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the date as the index
data.set_index('Date', inplace=True)

# a. Plot time series of daily unemployment rate
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Time Series of Daily Unemployment Rate')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.show()

# b. Test Stationarity
def test_stationarity(timeseries):
    # Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

test_stationarity(data['Unemployment Rate'])

# c. Construct ARIMA model
# Assuming 'Unemployment Rate' is the column containing unemployment rates
model = ARIMA(data['Unemployment Rate'], order=(5,1,0))  # Adjust order as needed
results = model.fit(disp=-1)

# Print the summary of the ARIMA model
print(results.summary())

# Plot ACF and PACF
plot_acf(data['Unemployment Rate'], lags=20)
plt.title('ACF')
plt.show()

# Optionally, plot PACF
# plot_pacf(data['Unemployment Rate'], lags=20)
# plt.title('PACF')
# plt.show()
