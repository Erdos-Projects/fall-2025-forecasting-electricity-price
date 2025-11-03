import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt

class SARIMA:

    def __init__(self, state, horizon=12, seasonal=True, max_order=5, max_p=2, max_q=2):
        self.state = state
        self.horizon = horizon
        self.seasonal = seasonal
        self.data_train = None
        self.model = None
        self.preds = None
        self.train_dates = None
        self.forecast_dates = None
        self.test_data = None
        self.max_order=max_order
        self.max_p = max_p
        self.max_q = max_q

    def fit(self, y: pd.Series, trace=True):
        self.data_train = y
        self.train_dates = y.index
        self.forecast_dates = pd.date_range(y.index[-1], periods=self.horizon+1, freq='MS')[1:]
        self.model = auto_arima(
            y, 
            seasonal=self.seasonal,
            m=self.horizon,
            trace=trace,
            error_action="ignore",
            suppress_warnings=True,
            max_order=self.max_order,
            max_q=self.max_q,
            max_p=self.max_p
        )

    def predict(self, test_data):
        self.test_data = test_data
        self.preds = self.model.predict(self.horizon)
        return self.preds

    def plot_forecast(self, save_path: str = None):
            # --- Plot Actual vs Forecast ---
        plt.figure(figsize=(8, 3.5))
        plt.plot(self.train_dates, self.data_train, label="Actual Price", color="blue")
        plt.plot(self.forecast_dates, self.preds, label="Forecast (SARIMA)", linestyle="--", color="red")
        plt.plot(self.forecast_dates, self.test_data, color='blue')

        # Draw a vertical line marking training cutoff
        plt.axvline(self.train_dates[-1], color="black", linestyle=":", label="Train Cutoff")

        plt.title(f"{self.state} — Actual vs Forecasted Residential Price")
        plt.xlabel("Date")
        plt.ylabel("Price (cents/kWh)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        if save_path:
            plt.savefig(save_path)



