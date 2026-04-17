import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("[INFO] Baseline grafikleri oluşturuluyor...")


# WAPE Hesaplama Fonksiyonu
def calculate_wape(y_true, y_pred):
    """
    Gerçek ve tahmin edilen değerler arasındaki WAPE'yi hesaplar.
    """
    total_abs_error = np.sum(np.abs(y_true - y_pred))
    total_actual = np.sum(y_true)

    if total_actual == 0:
        return 0.0  # Sıfıra bölme hatasını önlemek için

    wape = (total_abs_error / total_actual) * 100
    return wape


try:
    # 1. Verileri Yükle
    df_main = pd.read_csv('y_amazon-google-large.csv')
    df_main['ds'] = pd.to_datetime(df_main['ds'])

    b_pred = pd.read_csv('results/baseline_predictions.csv')
    b_pred['ds'] = pd.to_datetime(b_pred['ds'])

    # 2. İlk Kategori için Çizim
    first_id = b_pred['unique_id'].unique()[0]
    test_data = b_pred[b_pred['unique_id'] == first_id]

    # Geçmiş veriden son 3 günü (72 saat) bağlam olarak alalım
    train_data = df_main[(df_main['unique_id'] == first_id) & (df_main['ds'] < test_data['ds'].min())].tail(72)

    # 3. Dinamik Hata Hesaplamaları (Tüm test verisi üzerinden veya tek kategori üzerinden)
    # Burada tüm test verisi üzerinden global WAPE hesaplıyoruz.
    sn_wape = calculate_wape(b_pred['y'], b_pred['SeasonalNaive'])
    naive_wape = calculate_wape(b_pred['y'], b_pred['Naive'])

    print(f"[INFO] Hesaplanmış Seasonal Naive WAPE: %{sn_wape:.2f}")
    print(f"[INFO] Hesaplanmış Naive WAPE: %{naive_wape:.2f}")

    # 4. Grafik Çizimi
    plt.figure(figsize=(12, 6))

    # Gerçek Değerler
    plt.plot(train_data['ds'], train_data['y'], color='black', label='Geçmiş Veri (Bağlam)', linewidth=1)
    plt.plot(test_data['ds'], test_data['y'], color='black', linestyle='--', alpha=0.5, label='Gerçek Gelecek')

    # Modeller (Dinamik etiketlerle)
    plt.plot(test_data['ds'], test_data['SeasonalNaive'], color='green',
             label=f'Seasonal Naive (%{sn_wape:.1f} Hata)', linewidth=2.5)

    plt.plot(test_data['ds'], test_data['Naive'], color='orange',
             label=f'Naive (%{naive_wape:.1f} Hata)', linewidth=1.5)

    # Grafik Ayarları
    plt.title(f"Baseline Modellerin Tahmin Performansı (ID: {first_id})", fontsize=14)
    plt.xlabel("Zaman (Saatlik)")
    plt.ylabel("Yorum Sayısı")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Klasör yoksa oluştur
    os.makedirs('results', exist_ok=True)

    # Kaydet
    plt.savefig('results/baselines_forecast_plot.png', dpi=300)
    print("[INFO] 'results/baselines_forecast_plot.png' başarıyla kaydedildi!")

except Exception as e:
    print(f"[ERROR] Beklenmeyen bir hata oluştu: {e}")