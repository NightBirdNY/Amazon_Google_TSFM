import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from utilsforecast.losses import mae, rmse

print("N-BEATS Eğitimi Başlıyor...)")
start_time = time.time()

# 1. Veriyi Yükle
df = pd.read_csv('y_amazon-google-large.csv')
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

horizon = 24  # 24 saatlik tahmin ufku
lookback = 3 * horizon

# 2. N-BEATS Modelini Tanımla
model = NBEATS(
    h=horizon,
    input_size=lookback,
    stack_types=['trend', 'seasonality'],
    max_steps=500, # Hızlı eğitim.
    scaler_type='standard',
)

nf = NeuralForecast(models=[model], freq='H')

# 3. Model Eğitimi (Cross-Validation / Test ayrımı)
print("Model öğreniyor...")
crossval_df = nf.cross_validation(df=df, n_windows=1)

end_time = time.time()
training_time = end_time - start_time
print(f"Eğitim ve Tahmin tamamlandı! Süre: {training_time:.2f} saniye.")

# 4. Metrikleri Hesapla
# N-BEATS sonuçları crossval_df içinde geliyor
crossval_df = crossval_df.reset_index()

metrics_dict = {
    'Model': ['N-BEATS'],
    'Training_Time_Sec': [training_time],
    'MAE': [mae(crossval_df, models=['NBEATS'])['NBEATS'].mean()],
    'RMSE': [rmse(crossval_df, models=['NBEATS'])['NBEATS'].mean()],
    'WAPE': [abs(crossval_df['y'] - crossval_df['NBEATS']).sum() / (crossval_df['y'].sum() + 1e-5)]
}

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv('results/nbeats_metrics.csv', index=False)

# Daha sonra Chronos ile birleştirmek için tahminleri kaydet
crossval_df.rename(columns={'NBEATS': 'N-BEATS'}, inplace=True)
crossval_df[['unique_id', 'ds', 'y', 'N-BEATS']].to_csv('results/nbeats_predictions.csv', index=False)

# 5. Görselleştirme
first_uid = crossval_df['unique_id'].unique()[0]
plot_test = crossval_df[crossval_df['unique_id'] == first_uid]

# Geçmiş veriyi çizmek için ana veri setinden o kategoriyi alıyoruz
plot_train = df[(df['unique_id'] == first_uid) & (df['ds'] < plot_test['ds'].min())].tail(72)

plt.figure(figsize=(12, 6))
plt.plot(plot_train['ds'], plot_train['y'], label='Geçmiş (Bağlam)', color='black')
plt.plot(plot_test['ds'], plot_test['y'], label='Gerçek Gelecek', color='blue', linestyle='--')
plt.plot(plot_test['ds'], plot_test['N-BEATS'], label='N-BEATS Tahmini', color='green', linewidth=2)
plt.title(f"N-BEATS Tahmini vs Gerçek Değerler (ID: {first_uid})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/nbeats_forecast_plot.png')

print("="*50)
print(f"[INFO] N-BEATS Ortalama MAE: {metrics_dict['MAE'][0]:.4f}")
print(f"[INFO] N-BEATS WAPE: {metrics_dict['WAPE'][0]:.4f}")
print("="*50)