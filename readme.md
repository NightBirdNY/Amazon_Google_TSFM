# Zaman Serisi Temel Modellerinde Kalibrasyon ve Başarı Analizi (Beyond Accuracy)

Bu proje, zaman serisi tahminlemesinde geleneksel derin öğrenme modelleri (N-BEATS), klasik istatistiksel yöntemler (ARIMA) ve modern Zaman Serisi Temel Modellerinin (Amazon Chronos) karşılaştırmalı analizini içermektedir. Çalışmanın odak noktası, modellerin sadece noktasal doğruluğunu (Accuracy) değil, olasılıksal kalibrasyon ve "dürüstlük" metriklerini değerlendirmektir.

## 📊 Veri Seti
Kullanılan veri seti **UCI Amazon Reviews and Google Local** saatlik (hourly) verisidir. Veri, birbirinden bağımsız çeşitli e-ticaret ve lokasyon kategorilerindeki insan davranışının periyodik ritimlerini içermektedir.

## 🎯 Proje Hedefleri
- **Foundation Models vs Baselines:** Amazon Chronos'un (Zero-Shot) performansı ile spesifik eğitilmiş N-BEATS mimarisinin karşılaştırılması.
- **Olasılıksal Kalibrasyon Analizi:** Modellerin tahmin belirsizliğinin PCE (Probabilistic Calibration Error) ve CCE (Centered Calibration Error) gibi modern metriklerle ölçülmesi.
- **Ablasyon Çalışması:** N-BEATS mimarisinde bakış penceresi (lookback window) ve erken durdurma (early stopping) mekanizmalarının "Aşırı Özgüven" (Overconfidence) üzerindeki etkisinin incelenmesi.

## 🛠️ Kurulum ve Donanım
Gereksinimleri yüklemek için:
```bash
pip install -r requirements.txt