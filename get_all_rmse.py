import pandas as pd
import numpy as np

print("Kayıtlı tahmin dosyalarından RMSE değerleri hesaplanıyor...\n")

try:
    # 1. Tahmin dosyalarını yükle
    c_pred = pd.read_csv('results/chronos_predictions.csv')
    n_pred = pd.read_csv('results/nbeats_predictions.csv')
    b_pred = pd.read_csv('results/baseline_predictions.csv')


    # 2. RMSE Hesaplama Fonksiyonu (Kök Ortalama Kare Hata)
    def calculate_rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


    # 3. Tüm modeller için hesapla ve ekrana yazdır
    print(f"[INFO] N-BEATS RMSE       : {calculate_rmse(n_pred['y'], n_pred['N-BEATS']):.4f}")
    print(f"[INFO] Seasonal Naive RMSE: {calculate_rmse(b_pred['y'], b_pred['SeasonalNaive']):.4f}")
    print(f"[INFO] Chronos RMSE       : {calculate_rmse(c_pred['y'], c_pred['Chronos']):.4f}")
    print(f"[INFO] Naive (Düz) RMSE   : {calculate_rmse(b_pred['y'], b_pred['Naive']):.4f}")

except Exception as e:
    print(f"Dosyalar okunurken hata oluştu: {e}")