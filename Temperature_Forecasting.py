import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA

# Load the dataset
data = pd.read_csv('temperature_change.csv')  # Replace 'temperature_change.csv' with the actual file name

# Convert the date column to datetime format
data['DATE'] = pd.to_datetime(data['DATE'])

# Set the date as the index
data.set_index('DATE', inplace=True)

# a. Plot time series of monthly temperature change
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Time Series of Monthly Temperature Change')
plt.xlabel('Date')
plt.ylabel('Temperature Change')
plt.show()

# b. Test Stationarity
def test_stationarity(timeseries):
    # Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

test_stationarity(data['Temperature Change'])

# c. Construct ARIMA model
# Assuming 'Temperature Change' is the column containing temperature change readings
model = ARIMA(data['Temperature Change'], order=(5,1,0))  # Adjust order as needed
results = model.fit(disp=-1)

# Print the summary of the ARIMA model
print(results.summary())

# Plot ACF and PACF
plot_acf(data['Temperature Change'], lags=20)
plt.title('ACF')
plt.show()

# Optionally, plot PACF
# plot_pacf(data['Temperature Change'], lags=20)
# plt.title('PACF')
# plt.show()
