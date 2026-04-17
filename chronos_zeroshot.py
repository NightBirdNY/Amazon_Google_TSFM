import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from chronos import ChronosPipeline
from utilsforecast.losses import mae, rmse

# 1. Klasör Altyapısını Kur
os.makedirs('results', exist_ok=True)

# 2. Veriyi Yükle ve Hazırla
print("Veri yükleniyor ve Train/Test olarak ayrılıyor...")
df = pd.read_csv('y_amazon-google-large.csv')
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

horizon = 24  # 24 saatlik tahmin ufku

# Son 24 saati gizleyip (test), geri kalanını bağlam (train) olarak veriyoruz
df['reverse_rank'] = df.groupby('unique_id').cumcount(ascending=False)
train_df = df[df['reverse_rank'] >= horizon].drop(columns=['reverse_rank']).reset_index(drop=True)
test_df = df[df['reverse_rank'] < horizon].drop(columns=['reverse_rank']).reset_index(drop=True)

# 3. Chronos Modelini Yükle
model_name = 'amazon/chronos-t5-tiny'
print(f"{model_name} belleğe yükleniyor...")

pipeline = ChronosPipeline.from_pretrained(
    model_name,
    device_map="cuda",
)

# 4. Tahmin İçin Veriyi Tensörlere Çevir (Hafıza Optimizasyonlu)
print("Veriler tensörlere dönüştürülüyor...")
context_tensors = []
unique_ids = train_df['unique_id'].unique()

# Chronos için maksimum bağlam sınırı
max_context = 512

for uid in unique_ids:
    # Tüm geçmişi değil, sadece en son 512 saati alıyoruz
    series_data = train_df[train_df['unique_id'] == uid]['y'].values[-max_context:]
    context_tensors.append(torch.tensor(series_data, dtype=torch.float32))

# 5. Zero-Shot Çıkarım
print("Tahmin ediliyor...")
start_inference = time.time()

batch_size = 32
all_forecasts = []

for i in range(0, len(context_tensors), batch_size):
    print(f"İşleniyor: {i}/{len(context_tensors)}...")
    batch = context_tensors[i:i + batch_size]


    batch_samples = pipeline.predict(batch, prediction_length=horizon, num_samples=20)
    all_forecasts.append(batch_samples)

    # GPU önbelleğini temizle
    torch.cuda.empty_cache()

# Parça parça gelen tahminleri tek bir tensörde birleştir.
forecast_samples = torch.cat(all_forecasts, dim=0)

end_inference = time.time()
inference_time = end_inference - start_inference
print(f"Tahmin tamamlandı! Çıkarım Süresi: {inference_time:.2f} saniye.")

# 6. Tahminleri Rapor Formatına Dönüştür
forecast_means = forecast_samples.mean(dim=1).numpy() # 20 senaryonun noktasal ortalaması

predictions = []
for i, uid in enumerate(unique_ids):
    uid_test = test_df[test_df['unique_id'] == uid].copy()
    uid_test['Chronos'] = forecast_means[i]
    predictions.append(uid_test)

combined = pd.concat(predictions, ignore_index=True)

# 7. Rapor İçin Metrikleri Hesapla
metrics_dict = {
    'Model': [model_name],
    'Inference_Time_Sec': [inference_time],
    'MAE': [mae(combined, models=['Chronos'])['Chronos'].mean()],
    'RMSE': [rmse(combined, models=['Chronos'])['Chronos'].mean()],
    'WAPE': [abs(combined['y'] - combined['Chronos']).sum() / (combined['y'].sum() + 1e-5)]
}

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv('results/chronos_metrics.csv', index=False)
combined.to_csv('results/chronos_predictions.csv', index=False)

# 8. Rapor Görseli Çizimi
first_uid = unique_ids[0]
plot_train = train_df[train_df['unique_id'] == first_uid].tail(72) # Son 3 günü çiz (Bağlam)
plot_test = combined[combined['unique_id'] == first_uid]

plt.figure(figsize=(12, 6))
plt.plot(plot_train['ds'], plot_train['y'], label='Geçmiş (Bağlam)', color='black')
plt.plot(plot_test['ds'], plot_test['y'], label='Gerçek Gelecek', color='blue', linestyle='--')
plt.plot(plot_test['ds'], plot_test['Chronos'], label='Chronos Tahmini', color='red', linewidth=2)
plt.title(f"Chronos Zero-Shot Tahmini vs Gerçek Değerler (ID: {first_uid})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/chronos_forecast_plot.png')

print("="*50)
print(f"Tüm sonuçlar 'results/' klasörüne kaydedildi.")
print(f"Ortalama MAE: {metrics_dict['MAE'][0]:.4f}")
print(f"WAPE: {metrics_dict['WAPE'][0]:.4f}")
print("="*50)