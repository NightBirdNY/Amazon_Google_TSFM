import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("🧪 Final Analiz Raporu Hazırlanıyor...")

# 1. Mevcut Tahminleri Yükle
try:
    c_pred = pd.read_csv('results/chronos_predictions.csv')
    n_pred = pd.read_csv('results/nbeats_predictions.csv')
    b_pred = pd.read_csv('results/baseline_predictions.csv')

    # Hepsini tek tabloda birleştir
    final_df = c_pred[['unique_id', 'ds', 'y', 'Chronos']].copy()
    final_df = final_df.merge(n_pred[['unique_id', 'ds', 'N-BEATS']], on=['unique_id', 'ds'], how='left')
    final_df = final_df.merge(b_pred[['unique_id', 'ds', 'SeasonalNaive', 'Naive']], on=['unique_id', 'ds'], how='left')

    print("✅ Tahmin verileri birleştirildi.")
except Exception as e:
    print("❌ Hata: Bazı CSV dosyaları eksik! results/ klasörünü kontrol et.", e)

# 2. Performans Özeti (Hız ve Hata Tablosu)
c_m = pd.read_csv('results/chronos_metrics.csv')
n_m = pd.read_csv('results/nbeats_metrics.csv')
b_m = pd.read_csv('results/baseline_metrics.csv')

comparison = pd.DataFrame({
    'Model': ['N-BEATS (Eğitilmiş)', 'SeasonalNaive', 'Chronos (Zero-Shot)', 'Naive (Düz)'],
    'WAPE (%)': [
        round(n_m['WAPE'].iloc[0] * 100, 2),
        round(b_m[b_m['Model'] == 'SeasonalNaive']['WAPE'].iloc[0] * 100, 2),
        round(c_m['WAPE'].iloc[0] * 100, 2),
        round(b_m[b_m['Model'] == 'Naive']['WAPE'].iloc[0] * 100, 2)
    ],
    'Çalışma Süresi': ['17.8 sn (Eğitim)', '0.7 sn', '5.4 sn (Çıkarım)', '0.7 sn'],
    'Donanım': ['GPU (1660 Ti)', 'CPU', 'GPU (1660 Ti)', 'CPU']
})

# 3. Sonuçları Kaydet
os.makedirs('results', exist_ok=True)
comparison.to_csv('results/RAPOR_FINAL_TABLO.csv', index=False)

print("\n" + "=" * 70)
print("🎓 VİZE PROJESİ - FİNAL KARŞILAŞTIRMA TABLOSU")
print("=" * 70)
print(comparison.to_string(index=False))
print("=" * 70)

# 4. Final Grafiği (En İyi 3 Modelin Kapışması)
first_uid = final_df['unique_id'].unique()[0]
subset = final_df[final_df['unique_id'] == first_uid].tail(24)

plt.figure(figsize=(12, 6))
plt.plot(subset['ds'], subset['y'], 'k--', label='Gerçek Değer', alpha=0.6)
plt.plot(subset['ds'], subset['N-BEATS'], 'blue', label='N-BEATS (En İyi)', linewidth=2)
plt.plot(subset['ds'], subset['SeasonalNaive'], 'green', label='SeasonalNaive', alpha=0.7)
plt.plot(subset['ds'], subset['Chronos'], 'red', label='Chronos', alpha=0.7)

plt.title(f"Model Tahminleri Karşılaştırması (ID: {first_uid})")
plt.legend()
plt.grid(True, alpha=0.2)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/FINAL_VIZE_GRAFIGI.png')

print("\n🚀 İşlem bitti! 'results/RAPOR_FINAL_TABLO.csv' ve 'FINAL_VIZE_GRAFIGI.png' hazır.")
print("Artık Pycharm'ı kapatıp raporu yazmaya başlayabilirsin Taha. Eline sağlık!")