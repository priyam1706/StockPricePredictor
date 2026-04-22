# StockPricePredictor
# Stock Price Predictor

A deep learning project that forecasts stock closing prices using a GRU (Gated Recurrent Unit) neural network built with PyTorch. Includes time-series analysis, model training, and an interactive Streamlit dashboard for visualization.

## Features

- Fetches historical OHLCV data for AMZN, IBM, and MSFT using yFinance
- Time-series decomposition and stationarity testing (ADF Test) via Statsmodels
- GRU-based model trained with PyTorch for sequential price forecasting
- MinMax normalization and sequence windowing for model input preparation
- Interactive Streamlit dashboard to visualize predictions vs actuals
- Trained model weights saved as `.pth` files per ticker

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, GRU
- **Data:** yFinance, Pandas
- **Analysis:** Statsmodels, Scikit-learn
- **Visualization:** Streamlit, Matplotlib

## Model Performance

| Ticker | MAE | RMSE | R² |
|--------|-----|------|----|
| MSFT | 6.47 | 8.04 | 0.967 |
| AMZN | 67.24 | 72.94 | -0.33 |
| IBM | 126.88 | 145.90 | -1.48 |

> MSFT achieved the strongest results with R² of 0.967, indicating high predictive accuracy on test data.

## How to Run

**1. Install dependencies**
```bash
pip install -r requirement.txt
```

**2. Run the Streamlit app**
```bash
streamlit run main.py
```

**3. Or explore the notebook**
```bash
jupyter notebook main.ipynb
```

## Project Structure

```
StockPricePredictor/
├── main.py                  # Streamlit app + full pipeline
├── main.ipynb               # Jupyter notebook version
├── performance_metrics.csv  # Model evaluation results
├── stock_prices.csv         # Fetched historical data
├── MSFT_gru_model.pth       # Saved model weights - MSFT
├── AMZN_gru_model.pth       # Saved model weights - AMZN
├── IBM_gru_model.pth        # Saved model weights - IBM
└── requirement.txt          # Dependencies
```
