import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import joblib

# Load data
df = yf.download("AAPL", start="2010-01-01")

# Use only Close price
df = df[['Close']]

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

# Prepare training data
x_train, y_train = [], []

for i in range(100, len(scaled_data)):
    x_train.append(scaled_data[i-100:i])
    y_train.append(scaled_data[i])

x_train, y_train = np.array(x_train), np.array(y_train)

# Build model
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Save model + scaler
model.save("model.h5")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")