import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .st-b7 {
        color: #1f77b4;
    }
    .st-cg {
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox>div>div>div>div {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title('📈 AI Stock Market Predictor')
st.markdown("""
This app uses an LSTM neural network to predict stock prices. 
Select a stock ticker and date range to see predictions.
""")

# Sidebar for user inputs
with st.sidebar:
    st.header('User Input Parameters')
    
    # Ticker input with popular options
    ticker = st.selectbox(
        'Select Stock Ticker',
        ['NVDA', 'AAPL', 'BTC-USD'],
        index=0
    )
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime(2015, 1, 1),
            min_value=datetime(2000, 1, 1),
            max_value=datetime.today()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime(2025, 1, 1),
            min_value=datetime(2000, 1, 1),
            max_value=datetime.today()
        )
    
    # Model parameters
    st.subheader('Model Parameters')
    epochs = st.slider('Number of Epochs', 10, 200, 100)
    batch_size = st.slider('Batch Size', 16, 128, 32, 16)
    train_size = st.slider('Training Data Size (%)', 50, 90, 70)
    
    st.markdown("---")
    st.markdown("""
    *Note: First run may take longer as the model trains.
    Subsequent runs will be faster if using the same ticker.*
    """)

# Main function to run the app
def main():
    # Initialize session state for model caching
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.scaler = None
        st.session_state.trained_on = None
    
    # Fetch and prepare data
    @st.cache_data(show_spinner=False)
def get_data(ticker, start_date, end_date):
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return None
    try:
        with st.spinner(f'Fetching {ticker} data from Yahoo Finance...'):
            data = yf.download(ticker, start=start_date, end=end_date)
            if data is None or data.empty:
                st.error("No data found for this stock ticker and date range. Please try again.")
                return None
            return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

    
    data = get_data(ticker, start_date, end_date)
    
    if data is None:
        return
    
    # Show raw data
    with st.expander("View Raw Data"):
        st.dataframe(data.tail(10).style.format("{:.2f}"))
    
    # Basic statistics
    st.subheader('Basic Statistics')
    st.dataframe(data.describe().style.format("{:.2f}"))
    
    # Plot closing price
    st.subheader(f'{ticker} Closing Price History')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label='Close Price', color='#1f77b4')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Data preprocessing
    @st.cache_data(show_spinner=False)
    def preprocess_data(data, train_size):
        # Use only closing prices
        df = data[['Close']]
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)
        
        # Create training dataset
        training_size = int(len(scaled_data) * (train_size/100))
        train_data = scaled_data[0:training_size, :]
        
        # Create sequences for LSTM
        X_train = []
        y_train = []
        
        for i in range(100, len(train_data)):
            X_train.append(train_data[i-100:i, 0])
            y_train.append(train_data[i, 0])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        return X_train, y_train, scaler, training_size, scaled_data
    
    # Only show training button if new ticker or parameters changed
    if (st.session_state.trained_on != ticker or 
        st.session_state.model is None):
        
        if st.button('Train Model'):
            with st.spinner('Preparing data and training model...'):
                X_train, y_train, scaler, training_size, scaled_data = preprocess_data(data, train_size)
                
                # Build model
                def build_model():
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
                    return model
                
                model = build_model()
                
                # Train model with progress bar
                early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                
                progress_text = "Training in progress. Please wait..."
                progress_bar = st.progress(0)
                
                def on_epoch_end(epoch, logs):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(min(progress, 1.0))
                
                history = model.fit(
                    X_train, 
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                progress_bar.empty()
                
                # Save to session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.trained_on = ticker
                st.session_state.training_size = training_size
                st.session_state.scaled_data = scaled_data
                
                st.success('Model training completed!')
    
    # Prediction section
    if st.session_state.model is not None:
        st.subheader('Model Predictions')
        
        # Make predictions
        def make_predictions():
            model = st.session_state.model
            scaler = st.session_state.scaler
            training_size = st.session_state.training_size
            scaled_data = st.session_state.scaled_data
            
            # Prepare test data
            test_data = scaled_data[training_size - 100:, :]
            
            X_test = []
            y_test = scaled_data[training_size:, :]
            
            for i in range(100, len(test_data)):
                X_test.append(test_data[i-100:i, 0])
            
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # Make predictions
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            
            # Prepare data for plotting
            train = data[:training_size]
            valid = data[training_size:]
            valid['Predictions'] = predictions
            
            return train, valid
        
        train, valid = make_predictions()
        
        # Plot predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train['Close'], label='Training Data', color='#1f77b4')
        ax.plot(valid['Close'], label='Actual Price', color='#2ca02c')
        ax.plot(valid['Predictions'], label='Predicted Price', color='#ff7f0e', linestyle='--')
        ax.set_title(f'{ticker} Stock Price Prediction')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Close Price USD ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        
    #     # Show prediction metrics
    #     st.subheader('Prediction Metrics')
    #     col1, col2 = st.columns(2)
        
    #    # Drop rows with NaNs and Close == 0 to avoid divide by zero in MAPE
    #     valid_clean = valid[['Close', 'Predictions']].dropna()
    #     valid_clean = valid_clean[valid_clean['Close'] != 0]

    #     # Calculate RMSE and MAPE
    #     rmse = np.sqrt(np.mean((valid_clean['Close'] - valid_clean['Predictions'])**2))
    #     mape = np.mean(np.abs((valid_clean['Close'] - valid_clean['Predictions']) / valid_clean['Close'])) * 100

    #     # Display metrics
    #     col1.metric("RMSE (Root Mean Squared Error)", f"${rmse:.2f}")
    #     col2.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.2f}%")


        
        # Show prediction table
        with st.expander("View Prediction Data"):
            st.dataframe(valid.tail(10).style.format("{:.2f}"))
        
        # Download predictions
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        
        csv = convert_df(valid[['Close', 'Predictions']])
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f'{ticker}_predictions.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
