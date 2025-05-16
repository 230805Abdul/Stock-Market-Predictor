AI Stock Market Predictor
This is a Streamlit web application that uses an LSTM neural network to predict stock closing prices for selected tickers using historical data from Binance. It allows users to select a stock ticker, date range, and model parameters, and then trains a neural network to generate and visualize predictions.

 Features
 Interactive UI with Streamlit

 LSTM-based deep learning model for time-series forecasting

 Custom date selection for historical stock data

 Visualization of historical and predicted stock prices

 Download predictions as CSV

 Adjustable model parameters (epochs, batch size, training data split)

 Clean UI with custom CSS styling

 Session state caching to avoid retraining for the same ticker

Built With
Streamlit
TensorFlow / Keras
Scikit-learn
Python-Binance
Matplotlib
NumPy
Pandas

Install dependencies:
It's recommended to use a virtual environment.
pip install -r requirements.txt

requirements.txt should include:
streamlit
numpy
pandas
matplotlib
python-binance
scikit-learn
tensorflow

Run the app:
streamlit run app_LSTM.py

How It Works:
User selects a stock ticker, date range, and model parameters.

Historical stock data is fetched using binance.

Closing prices are scaled and converted into 100-timestep sequences for LSTM input.

The model is trained on the training portion of the data.

Predictions are made on the test portion and plotted alongside actual values.

Users can view raw prediction data and download results as CSV.

