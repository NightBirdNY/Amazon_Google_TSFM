import pandas as pd
import matplotlib.pyplot as plt
import os

print("📊 Baseline grafikleri oluşturuluyor...")

# 1. Verileri Yükle
try:
    df_main = pd.read_csv('y_amazon-google-large.csv')
    df_main['ds'] = pd.to_datetime(df_main['ds'])

    b_pred = pd.read_csv('results/baseline_predictions.csv')
    b_pred['ds'] = pd.to_datetime(b_pred['ds'])

    # 2. İlk Kategori (ID: 0) için Çizim
    first_id = b_pred['unique_id'].unique()[0]
    test_data = b_pred[b_pred['unique_id'] == first_id]
    # Geçmiş veriden son 3 günü (72 saat) bağlam olarak alalım
    train_data = df_main[(df_main['unique_id'] == first_id) & (df_main['ds'] < test_data['ds'].min())].tail(72)

    # 3. Grafik Çizimi
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['ds'], train_data['y'], color='black', label='Geçmiş Veri (Bağlam)', linewidth=1)
    plt.plot(test_data['ds'], test_data['y'], color='black', linestyle='--', alpha=0.5, label='Gerçek Gelecek')

    # Modeller
    plt.plot(test_data['ds'], test_data['SeasonalNaive'], color='green', label='Seasonal Naive (%14.6 Hata)',
             linewidth=2.5)
    plt.plot(test_data['ds'], test_data['Naive'], color='orange', label='Naive (%52.0 Hata)', linewidth=1.5)

    plt.title(f"Baseline (Temel) Modellerin Tahmin Performansı (ID: {first_id})", fontsize=14)
    plt.xlabel("Zaman (Saatlik)")
    plt.ylabel("Yorum Sayısı")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Kaydet
    plt.savefig('results/baselines_forecast_plot.png', dpi=300)
    print("✅ 'results/baselines_forecast_plot.png' başarıyla kaydedildi!")

except Exception as e:
    print(f"❌ Hata oluştu: {e}")