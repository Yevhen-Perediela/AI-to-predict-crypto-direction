import pandas as pd
import requests
import time
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta

def get_klines(symbol="BTCUSDT", interval="15m", limit=1000, start_time=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = int(start_time)

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_multiple_klines(symbol="BTCUSDT", interval="15m", total_limit=10000):
    all_data = []
    start = datetime.utcnow() - timedelta(minutes=150000 + 60)
    last_time = int(start.timestamp() * 1000)

    while len(all_data) < total_limit:
        limit = min(1000, total_limit - len(all_data))
        data = get_klines(symbol, interval, limit=limit, start_time=last_time)

        if not data:
            break

        all_data.extend(data)
        last_time = data[-1][0] + 1 
        time.sleep(0.1)

    return all_data

def save_to_csv(symbol="BTCUSDT", interval="15m", filename="dane.csv"):
    raw_data = fetch_multiple_klines(symbol, interval, total_limit=10000)
    
    df = pd.DataFrame(raw_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi.rsi()

    df = df[["timestamp", "open", "high", "low", "close", "volume", "rsi"]]
    df.dropna(inplace=True)
    df.to_csv(filename, index=False)
    print(f"Zapisano {len(df)} świec do pliku: {filename}")

if __name__ == "__main__":
    save_to_csv(symbol="BTCUSDT", interval="15m")
