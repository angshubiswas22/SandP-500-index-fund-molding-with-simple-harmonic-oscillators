import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define 2-Component Harmonic Cosine Equation
def shm_2component_model(t, A1, w1, p1, A2, w2, p2, C):
    c1 = A1 * np.cos(w1 * t + p1)
    c2 = A2 * np.cos(w2 * t + p2)
    return c1 + c2 + C

def main():
    print("=" * 70)
    print(" METHOD 1: DIRECT OPTIMIZATION - 2-COMPONENT HARMONIC MODEL")
    print("=" * 70)

    # 1. Load S&P 500 Dataset
    csv_path = 'sp500_index.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file '{csv_path}' not found.")

    df = pd.read_csv(csv_path)
    if df.columns[0].startswith('('):
        df.columns = ['date', 'close', 'high', 'low', 'open', 'volume', 'ticker']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Loaded {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}.")

    # 2. Chronological 75:25 Train-Test Split (Scenario B)
    train_size = int(len(df) * 0.75)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    t_all = np.arange(len(df), dtype=float)
    t_train = t_all[:train_size]
    t_test = t_all[train_size:]

    y_train_raw = train_df['close'].values
    y_test_raw = test_df['close'].values

    # 3. Log-Linear Detrending (Fit ON TRAIN ONLY to prevent data leakage)
    y_train_log = np.log(y_train_raw)
    y_test_log = np.log(y_test_raw)

    m, c = np.polyfit(t_train, y_train_log, deg=1)

    trend_train_log = m * t_train + c
    trend_test_log = m * t_test + c

    y_train_detrended = y_train_log - trend_train_log
    y_test_detrended = y_test_log - trend_test_log

    # 4. Target Scaling
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_detrended.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test_detrended.reshape(-1, 1)).flatten()

    # 5. FFT Frequency Initialization (Extract Top 2 Frequencies)
    fft_vals = np.fft.rfft(y_train_scaled)
    fft_freqs = np.fft.rfftfreq(len(y_train_scaled))
    non_zero_idx = np.where(fft_freqs > 0)[0]
    top_2_idx = non_zero_idx[np.argsort(np.abs(fft_vals[non_zero_idx]))[-2:][::-1]]
    init_omegas = 2 * np.pi * fft_freqs[top_2_idx]

    p0 = [
        1.0, init_omegas[0], 0.0,
        0.8, init_omegas[1], 0.0,
        0.0
    ]

    # Parameter Bounds (2 * [Amplitude, Frequency, Phase] + Offset)
    lower_bounds = [0.0, 1e-6, -np.pi] * 2 + [-5.0]
    upper_bounds = [10.0, np.pi, np.pi] * 2 + [5.0]
    bounds = (lower_bounds, upper_bounds)

    print("\nExecuting Non-Linear Curve Fitting (7 Parameters)...")
    popt, _ = curve_fit(
        shm_2component_model,
        t_train,
        y_train_scaled,
        p0=p0,
        bounds=bounds,
        maxfev=50000
    )

    # 6. Predict & Invert Transformations
    pred_train_scaled = shm_2component_model(t_train, *popt)
    pred_test_scaled = shm_2component_model(t_test, *popt)

    pred_train_detrended = scaler_y.inverse_transform(pred_train_scaled.reshape(-1, 1)).flatten()
    pred_test_detrended = scaler_y.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()

    pred_train_raw = np.exp(pred_train_detrended + trend_train_log)
    pred_test_raw = np.exp(pred_test_detrended + trend_test_log)

    # 7. Metrics
    print("\n" + "=" * 50)
    print(" EVALUATION METRICS (Method 1: 2-Component Model)")
    print("=" * 50)
    print(f" Train RMSE: ${np.sqrt(mean_squared_error(y_train_raw, pred_train_raw)):.2f} | Test RMSE: ${np.sqrt(mean_squared_error(y_test_raw, pred_test_raw)):.2f}")
    print(f" Train MAE:  ${mean_absolute_error(y_train_raw, pred_train_raw):.2f} | Test MAE:  ${mean_absolute_error(y_test_raw, pred_test_raw):.2f}")
    print(f" Train R2:   {r2_score(y_train_raw, pred_train_raw):.4f}   | Test R2:   {r2_score(y_test_raw, pred_test_raw):.4f}")

    # 8. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'].iloc[:train_size], y_train_raw, label='Actual Price (Train)', color='#1f77b4', alpha=0.8)
    plt.plot(df['date'].iloc[train_size:], y_test_raw, label='Actual Price (Test)', color='#ff7f0e', alpha=0.8)
    plt.plot(df['date'].iloc[:train_size], pred_train_raw, label='2-Component Model Fit (Train)', color='#2ca02c', linestyle='--')
    plt.plot(df['date'].iloc[train_size:], pred_test_raw, label='2-Component Model Forecast (Test)', color='#d62728', linestyle='--')
    plt.title('S&P 500 Price Forecast - Method 1: 2-Component curve_fit', fontsize=12, fontweight='bold')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('method1_2component_results.png', dpi=300)
    print("\nPlot saved as 'method1_2component_results.png'.")

if __name__ == '__main__':
    main()