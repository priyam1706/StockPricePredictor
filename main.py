#!/usr/bin/env python
# coding: utf-8

# # Steps:
# 1. Data collection and preprocessing.
# 2. Time series analysis (trends, seasonality, stationarity).
# 3. Data preparation for GRU modeling.
# 4. Building and training the GRU model in PyTorch.
# 5. Evaluating the model and forecasting.

# In[2]:

import os
os.environ["TORCH_DISABLE_PATHS_EXTRACT"] = "1"


import yfinance as yf #fetch the historical stock market data from yahoo finance; use to fetch the data 
import pandas as pd  #data manipulation and analysis ( It simplifies handling tabular data)

stocks = ['AMZN', 'IBM', 'MSFT'] #only 3 companies to be fetch data
data = {}  

#download the stocks data from yf api 
for stock in stocks:
    data[stock] = yf.download(stock, start='2015-01-01', end='2025-04-29')['Close']

# Combines the closing prices of all stocks into a single DataFrame.
df = pd.concat(data, axis=1)
df.columns = stocks  #Rename the column
df.to_csv('stock_prices.csv') #exporting to CSV


# In[21]:

import streamlit as st
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for stock in stocks:
    plt.plot(df[stock], label=stock)
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.grid(True)
plt.ylabel('Close Price (USD)')
plt.legend()
st.pyplot(plt)


# In[19]:

import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose

for stock in stocks:
    decomposition = seasonal_decompose(df[stock], model='multiplicative', period=252)  # Approx. trading days in a year

    # Extract components
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot the decomposition
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.grid(True)
    plt.title(f'Decomposition of {stock}')
    st.pyplot(plt)

    # Display components
    print(f'{stock} Trend Component:')
    print(trend.dropna().head())
    print(f'\n{stock} Seasonal Component:')
    print(seasonal.dropna().head())
    print(f'\n{stock} Residual Component:')
    print(residual.dropna().head())


# In[15]:


from statsmodels.tsa.stattools import adfuller

for stock in stocks:
    result = adfuller(df[stock].dropna())
    print(f'{stock} ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print(f'{stock} is non-stationary. Consider differencing.')
    else:
        print(f'{stock} is stationary.')


# In[5]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = {}
for stock in stocks:
    scaled_data[stock] = scaler.fit_transform(df[stock].values.reshape(-1, 1))


# In[6]:


import numpy as np

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = {}, {}
for stock in stocks:
    X[stock], y[stock] = create_sequences(scaled_data[stock], seq_length)


# In[7]:


train_size = int(len(X[stocks[0]]) * 0.8)
X_train, X_test, y_train, y_test = {}, {}, {}, {}
for stock in stocks:
    X_train[stock] = X[stock][:train_size]
    X_test[stock] = X[stock][train_size:]
    y_train[stock] = y[stock][:train_size]
    y_test[stock] = y[stock][train_size:]


# In[8]:


import torch

for stock in stocks:
    X_train[stock] = torch.FloatTensor(X_train[stock])
    X_test[stock] = torch.FloatTensor(X_test[stock])
    y_train[stock] = torch.FloatTensor(y_train[stock])
    y_test[stock] = torch.FloatTensor(y_test[stock])


# In[9]:


import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# In[10]:


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(device))
            test_loss = criterion(test_outputs, y_test.to(device))

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    return model

# Train a model for each stock
models = {}
for stock in stocks:
    model = GRUModel()
    print(f'\nTraining model for {stock}...')
    models[stock] = train_model(model, X_train[stock], y_train[stock], X_test[stock], y_test[stock])


# In[11]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

for stock in stocks:
    model = models[stock]
    model.eval()
    with torch.no_grad():
        predictions = model(X_test[stock]).numpy()
        actual = y_test[stock].numpy()

    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    print(f'{stock} - MSE: {mse:.4f}, MAE: {mae:.4f}')


# In[17]:


def forecast_next_day(model, last_sequence, scaler):
    model.eval()
    with torch.no_grad():
        last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0)
        pred = model(last_sequence).numpy()
        pred = scaler.inverse_transform(pred)
    return pred[0, 0]

for stock in stocks:
    last_sequence = scaled_data[stock][-seq_length:]
    next_price = forecast_next_day(models[stock], last_sequence, scaler)
    print(f'Predicted next day price for {stock}: ${next_price:.2f}')


# In[18]:

import streamlit as st
from sklearn.metrics import mean_squared_error
import numpy as np

for stock in stocks:
    model = models[stock]
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test[stock]).numpy()
        pred = scaler.inverse_transform(pred_scaled)
        actual = scaler.inverse_transform(y_test[stock].numpy())

    rmse = np.sqrt(mean_squared_error(actual, pred))

    dates = df.index[-len(actual):]  # Align with actual dates

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, pred, label='Predicted')
    plt.title(f'{stock} Stock Price Prediction (RMSE: {rmse:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
