import pandas as pd
import numpy as np
import warnings
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning,
    module="sklearn"
)


class RandomForrestForecaster:

    def __init__(self, state, n_componets=5, horizon=12, n_estimators=100, random_state=42):
        self.n_components = n_componets
        self.horizon = horizon
        self.preds = None
        self.y_train = None
        self.X_train = None
        self.pca_forecast = None
        self.X_pca = None
        self.pca_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=n_componets))])
        self.rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.state = state

    def fit(self, X: pd.DataFrame, y: pd.Series, trace: bool=False):
        self.y_train = y
        self.X_train = X
        self.train_dates = y.index
        self.forecast_dates = pd.date_range(y.index[-1], periods=self.horizon+1, freq='MS')[1:]
        self.X_pca = self.pca_pipe.fit_transform(X)
        self.rf_model.fit(self.X_pca, y)

        self._get_PCA_Forecast(trace=trace)

    def predict(self):
        self.preds = self.rf_model.predict(self.pca_forecast)
        return self.preds
    

    def _get_PCA_Forecast(self, trace=False):
        self.X_pca = self.pca_pipe.fit_transform(self.X_train)
        self.pca_forecast = np.zeros(shape=(self.horizon, self.n_components))
        self.pca_sarimas = {}
        for i in tqdm(range(self.n_components)):
            if trace:
                print(f'forcasting PCA Component {i+1}')
            y = self.X_pca[:,i]
            sarima = auto_arima(
                y, 
                seasonal=True,
                m=self.horizon,
                error_action='ignore',
                max_q = 3,
                max_p = 2,
                max_order = 5,
                trace=trace
            )
            start = len(self.y_train) - self.horizon
            self.pca_forecast[:, i] = sarima.predict(self.horizon, start=start)
            self.pca_sarimas[f'component {i+1}'] = sarima
    
    def plot_forecast(self, test_data=None, save_path: str = None):
            # --- Plot Actual vs Forecast ---
        plt.figure(figsize=(8, 3.5))
        plt.plot(self.train_dates, self.y_train, label="Actual Price", color="blue")
        plt.plot(self.forecast_dates, self.preds, label="Forecast (RandomForrestForecaster)", linestyle="--", color="red")
        if test_data is not None:
            plt.plot(self.forecast_dates, test_data, color='blue')

        # Draw a vertical line marking training cutoff
        plt.axvline(self.train_dates[-1], color="black", linestyle=":", label="Train Cutoff")

        plt.title(f"{self.state} — Actual vs Forecasted Residential Price")
        plt.xlabel("Date")
        plt.ylabel("Price (cents/kWh)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    