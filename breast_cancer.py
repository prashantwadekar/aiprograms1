import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv"
data = pd.read_csv(url)

# a. Exploratory Data Analysis
print("a. Exploratory Data Analysis:")
print(data.head())
print("\nSummary Statistics:")
print(data.describe())
print("\nClass Distribution:")
print(data['Classification'].value_counts())

# b. Normalization and Sampling
X = data.drop(columns=['Classification'])
y = data['Classification']

# Standardize the feature values
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# c. Construct a Neural Network model for prediction
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict_classes(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
