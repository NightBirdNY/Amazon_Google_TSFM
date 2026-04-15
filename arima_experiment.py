import pandas as pd
import time
import os
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, Naive
from utilsforecast.losses import mae, rmse

print("📊 Basit Baselineler Başlatılıyor (CPU Dostu)...")
start_time = time.time()

# 1. Veriyi Yükle
df = pd.read_csv('y_amazon-google-large.csv')
df['ds'] = pd.to_datetime(df['ds'])

horizon = 24

# 2. Modelleri Tanımla (Hiçbiri bilgisayarı yormaz)
models = [
    SeasonalNaive(season_length=24), # Mevsimsel (24 saatlik döngü)
    Naive() # Saf (Son değeri tekrar et)
]

sf = StatsForecast(
    models=models,
    freq='H',
    n_jobs=1 # Tek çekirdek yeterli, risk almayalım
)

# 3. Tahmin Üret
print("🧐 İstatistiksel tahminler üretiliyor...")
cv_df = sf.cross_validation(df=df, h=horizon, n_windows=1)
cv_df = cv_df.reset_index()

end_time = time.time()
exec_time = end_time - start_time
print(f"✅ İşlem tamamlandı! Süre: {exec_time:.2f} saniye.")

# 4. Metrikleri Kaydet (Rapor için)
os.makedirs('results', exist_ok=True)
metrics_list = []

for model in ['SeasonalNaive', 'Naive']:
    m_dict = {
        'Model': [model],
        'Execution_Time_Sec': [exec_time / 2],
        'MAE': [mae(cv_df, models=[model])[model].mean()],
        'WAPE': [abs(cv_df['y'] - cv_df[model]).sum() / (cv_df['y'].sum() + 1e-5)]
    }
    metrics_list.append(pd.DataFrame(m_dict))

final_baseline_metrics = pd.concat(metrics_list)
final_baseline_metrics.to_csv('results/baseline_metrics.csv', index=False)

# Tahminleri sakla
cv_df.to_csv('results/baseline_predictions.csv', index=False)

print("\n🏆 BASELINE SONUÇLARI 🏆")
print(final_baseline_metrics)