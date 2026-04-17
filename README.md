# Zaman Serisi Tahminlemede Derin Öğrenme ve Temel Modellerin Karşılaştırmalı Analizi

Bu proje, yüksek frekanslı ve gürültülü (noisy) saatlik zaman serisi tahminlemesinde; özelleştirilmiş derin öğrenme mimarileri (N-BEATS), önceden eğitilmiş büyük Zaman Serisi Temel Modelleri (Amazon Chronos) ve geleneksel istatistiksel baz modellerin (Seasonal Naive) karşılaştırmalı analizini içermektedir.

## A. Veri Seti
Kullanılan veri seti **UCI Amazon Product and Google Locations Reviews** saatlik (hourly) verisidir. Veri seti, sıklıkla sıfır değerleri içeren (sparse) ve günün belirli saatlerinde yüksek varyans gösteren, insan davranışının periyodik ritimlerini barındıran zorlu bir yapıya sahiptir.

## B. Proje Hedefleri
- **Foundation Models vs Fine-Tuned Models:** Hiç eğitim almadan sıfır-atış (Zero-Shot) yapan Amazon Chronos ile veri setine özel olarak eğitilen N-BEATS mimarisinin performans kıyaslaması.
- **Baseline Kritikliği:** Klasik ARIMA modelinin saatlik yüksek gürültülü verilerde matematiksel yakınsama (convergence) hataları vererek çökmesi üzerine, en dürüst kıyaslama noktası olarak `Seasonal Naive` modelinin kullanılması ve verideki 24 saatlik periyodik döngünün gücünün ispatlanması.
- **Operasyonel Verimlilik ve Donanım Analizi:** Modellerin noktasal doğruluklarının (WAPE, RMSE) yanı sıra eğitim/çıkarım süreleri ve donanım (VRAM/CPU) maliyetlerinin incelenmesi.


## C. Kurulum ve Donanım
Deneyler, CUDA hızlandırması ile yerel bir Linux ortamında **NVIDIA GTX 1660 Ti (6GB VRAM)** kullanılarak gerçekleştirilmiştir. Chronos'un çıkarım aşamasında OOM (Out of Memory) hatalarını önlemek için *batching* (`batch_size=32`) ve *context limiting* (512 saat) stratejileri uygulanmıştır.

Kullanılan temel kütüphaneler (`NeuralForecast`, `StatsForecast`, `PyTorch`) ve diğer gereksinimleri yüklemek için:
```bash
pip install -r requirements.txt
```

## D. Ana Bulgular ve Model Performansları

Modellerin 24 saatlik tahmin ufku üzerindeki noktasal hata metrikleri (Accuracy) aşağıda sunulmuştur:

| Model | WAPE (%) | Global RMSE | Süre (Donanım) |
| :--- | :---: | :---: | :--- |
| **N-BEATS** (Eğitilmiş) | **%14.47** | **20.71** | 17.8 sn (1660 Ti) |
| **Seasonal Naive** (Baseline) | %14.66 | 21.90 | 0.7 sn (CPU) |
| **Chronos** (Zero-Shot) | %15.29 | 24.28 | 5.4 sn (1660 Ti) |
| **Naive** (Düz) | %52.02 | 84.22 | 0.7 sn (CPU) |

## E. Olasılıksal Kalibrasyon ve Güvenilirlik (Beyond Accuracy)

TSFM'lerin (Zaman Serisi Temel Modelleri) tahminlerinde *aşırı özgüvenli (overconfident)* olma sorununu analiz etmek adına modellerin %90 güven aralıkları (Prediction Intervals) çıkarılmış ve CCE/PCE skorları hesaplanmıştır:

| Model | Ampirik Kapsama (Hedef: %90) | PCE Skoru (↓) | CCE Skoru (↓) | SIW Skoru (↓) |
| :--- | :---: | :---: | :---: | :---: |
| **N-BEATS** | %95.71 | **0.0571** | **0.0812** | **1.4694** |
| **Chronos (TSFM)** | %96.88 | 0.0688 | 0.1034 | 1.7397 |

*Sonuç:* Chronos modeli, veriyi hiç görmemesine rağmen temkinli (geniş aralıklı) ve güvenilir bir profil sergilemiştir. Ancak N-BEATS mimarisi en dar aralığı üreterek (SIW: 1.46) belirsizliği çok daha yüksek bir dürüstlükle (CCE: 0.0812) yönetmeyi başarmıştır.

## F. Çalıştırma Sırası (Usage)

Deneyleri tekrarlamak veya metrikleri incelemek için sırasıyla şu adımları izleyebilirsiniz:
1. `nbeats_experiment.py` / `chronos_zeroshot.py` / `arima_experiment.py` 
2. `get_all_rmse.py` 
3. `calibration_metrics.py` 
4. `plot_baselines.py` 
