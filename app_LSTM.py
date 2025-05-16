import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from binance.client import Client
import os

# Initialize Binance Client 
client = Client()

# Set page config
st.set_page_config(
    page_title="Crypto Predictor with Binance",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .stProgress > div > div > div > div { background-color: #1f77b4; }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title(' Crypto Price Predictor (Binance)')
st.markdown("This app uses an LSTM neural network to predict crypto prices using Binance data.")

# Sidebar
with st.sidebar:
    st.header('Settings')

    symbol = st.selectbox(
        'Choose Cryptocurrency Pair (Binance Symbol)',
        ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
        index=0
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2024, 12, 31))

    st.subheader('Model Parameters')
    epochs = st.slider('Epochs', 10, 200, 50)
    batch_size = st.slider('Batch Size', 16, 128, 32, 16)
    train_size = st.slider('Train/Test Split (%)', 50, 90, 70)

# Function to get Binance historical data
@st.cache_data(show_spinner=False)
def get_binance_data(symbol, start_date, end_date, interval='1d'):
    start_str = start_date.strftime('%d %b %Y')
    end_str = end_date.strftime('%d %b %Y')

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
        end_str=end_str
    )

    if not klines:
        return None

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base_volume', 'Taker_buy_quote_volume', 'Ignore'
    ])

    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    return df

# Get data
data = get_binance_data(symbol, start_date, end_date)

if data is None or data.empty:
    st.error("No data returned. Try different dates or symbol.")
    st.stop()

# Show data
with st.expander(" View Raw Data"):
    st.dataframe(data.tail(10).style.format("{:.2f}"))

# Plot data
st.subheader(f"{symbol} Closing Price History")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Close'], label='Close Price', color='#1f77b4')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USDT)')
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# Preprocess data
def preprocess_data(data, train_size):
    df = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    training_size = int(len(scaled_data) * (train_size / 100))
    train_data = scaled_data[0:training_size]

    X_train, y_train = [], []
    for i in range(100, len(train_data)):
        X_train.append(train_data[i-100:i, 0])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    return X_train, y_train, scaler, training_size, scaled_data

if st.button('Train Model'):
    X_train, y_train, scaler, training_size, scaled_data = preprocess_data(data, train_size)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, callbacks=[early_stop])
        progress_bar.progress((epoch+1)/epochs)

    st.success("✅ Model training complete!")

    # Prediction
    test_data = scaled_data[training_size - 100:, :]
    X_test, y_test = [], scaled_data[training_size:, :]

    for i in range(100, len(test_data)):
        X_test.append(test_data[i-100:i, 0])

    X_test = np.array(X_test).reshape(len(X_test), 100, 1)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    train = data[:training_size]
    valid = data[training_size:]
    valid['Predictions'] = predictions

    # Plot predictions
    st.subheader(" Predicted vs Actual Price")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(train['Close'], label='Training Data', color='#1f77b4')
    ax2.plot(valid['Close'], label='Actual Price', color='#2ca02c')
    ax2.plot(valid['Predictions'], label='Predicted Price', color='#ff7f0e', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Close Price (USDT)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Download predictions
    csv = valid[['Close', 'Predictions']].to_csv().encode('utf-8')
    st.download_button(
        label="⬇ Download Predictions as CSV",
        data=csv,
        file_name=f"{symbol}_predictions.csv",
        mime='text/csv'
    )

    # View table
    with st.expander(" View Prediction Data"):
        st.dataframe(valid.tail(10).style.format("{:.2f}"))
