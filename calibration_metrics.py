import pandas as pd
import numpy as np

print("Kalibrasyon (PCE & SIW) ve Aşırı Özgüven Analizi Başlıyor...\n")

try:
    # 1. Tahminleri yükle
    c_pred = pd.read_csv('results/chronos_predictions.csv')
    n_pred = pd.read_csv('results/nbeats_predictions.csv')
    b_pred = pd.read_csv('results/baseline_predictions.csv')


    # 2. Kalibrasyon Hesaplama Fonksiyonu
    def calculate_calibration(y_true, y_pred, model_name):
        residuals = y_true - y_pred
        std_res = np.std(residuals)

        # %90 Güven Aralığı (Z-score = 1.645)
        lower_bound = y_pred - 1.645 * std_res
        upper_bound = y_pred + 1.645 * std_res

        # Ampirik Kapsama
        empirical_coverage = ((y_true >= lower_bound) & (y_true <= upper_bound)).mean()

        # Beklenen Kapsama
        expected_coverage = 0.90

        # PCE (Probability Calibration Error)
        pce = abs(empirical_coverage - expected_coverage)

        # SIW (Scaled Interval Width)
        width = upper_bound - lower_bound
        siw = np.mean(width) / (np.mean(y_true) + 1e-5)

        # Overconfidence Analizi (Model dar aralık verip yanılıyor mu?)
        is_overconfident = empirical_coverage < expected_coverage

        print(f"Model: {model_name}")
        print(f"- Ampirik Kapsama: %{empirical_coverage * 100:.2f} (Hedef: %90)")
        print(f"- PCE Skoru: {pce:.4f}")
        print(f"- SIW Skoru: {siw:.4f}")
        print(
            f"- Durum: {'AŞIRI ÖZGÜVENLİ' if is_overconfident else 'GÜVENİLİR'}\n")

        return pce, siw


    calculate_calibration(n_pred['y'], n_pred['N-BEATS'], "N-BEATS")
    calculate_calibration(c_pred['y'], c_pred['Chronos'], "Chronos")
    calculate_calibration(b_pred['y'], b_pred['SeasonalNaive'], "Seasonal Naive")

except Exception as e:
    print("Hata:", e)