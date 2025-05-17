import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator


df = pd.read_csv("dane.csv")

df["price_change"] = df["close"] - df["open"]
df["high_low_diff"] = df["high"] - df["low"]
df["volatility"] = df["high"] / df["low"]
df["rsi_diff"] = df["rsi"] - df["rsi"].shift(1)
df["ema_10"] = EMAIndicator(close=df["close"], window=10).ema_indicator()
df["macd"] = MACD(close=df["close"]).macd_diff()

# target
df["future_return"] = (df["close"].shift(-5) - df["close"]) / df["close"]
df["direction"] = (df["future_return"] > 0.002).astype(int)

df.dropna(inplace=True)

features = ["open", "high", "low", "close", "volume", "rsi",
            "price_change", "high_low_diff", "volatility", "rsi_diff",
            "ema_10", "macd"]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

SEQUENCE_LENGTH = 60
X, y = [], []

for i in range(SEQUENCE_LENGTH, len(df_scaled)):
    X.append(df_scaled[i - SEQUENCE_LENGTH:i])
    y.append(df["direction"].iloc[i])

X, y = np.array(X), np.array(y)

# podzial
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)


model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nDokładność LSTM (target 5 świec, +0.2%): {accuracy * 100:.2f}%")

last_sequence = df_scaled[-SEQUENCE_LENGTH:]
last_sequence = np.expand_dims(last_sequence, axis=0)

pred = model.predict(last_sequence)[0][0]
print("\nPredykcja na kolejną świecę (czy wzrost o 0.2% w 5 świec):")
print("🟢 GÓRA" if pred > 0.5 else "🔴 DÓŁ")
