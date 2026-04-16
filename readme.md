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