import os
import librosa
import numpy as np
import json
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model


def extract_features(dosya_yolu, hedef_uzunluk=None, n_mfcc=40, n_fft=2048, hop_length=512):
    """
    Ses dosyasından özellikler çıkarır
    """
    try:
        # Ses dosyasını yükle
        y, sr = librosa.load(dosya_yolu, sr=None)

        # Eğer dosya çok kısaysa (0.5 saniyeden az), bu dosyayı atla
        if len(y) / sr < 0.5:
            print(f"Uyarı: {dosya_yolu} çok kısa, atlanıyor.")
            return None

        # Hedef uzunluğa göre sesi kesme veya padding yapma
        orijinal_uzunluk = len(y) / sr
        if hedef_uzunluk is not None:
            if orijinal_uzunluk > hedef_uzunluk:
                # Rastgele bir kesit al (test için ortadan alalım)
                baslangic = (orijinal_uzunluk - hedef_uzunluk) / 2
                y = y[int(baslangic * sr):int((baslangic + hedef_uzunluk) * sr)]
            elif orijinal_uzunluk < hedef_uzunluk:
                # Kısa ses dosyalarını 0 ile doldur (padding)
                padding = np.zeros(int(hedef_uzunluk * sr) - len(y))
                y = np.concatenate((y, padding))

        # Sessizliği temizle
        y, _ = librosa.effects.trim(y, top_db=20)

        # Özellik çıkarma işlemi
        features = []

        # 1. MFCC (Mel Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        features.extend(mfccs_scaled)

        # 2. Spektral Merkezlik (Spectral Centroid)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))

        # 3. Spektral Bant Genişliği (Spectral Bandwidth)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))

        # 4. Sıfır Geçiş Oranı (Zero Crossing Rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zero_crossing_rate))
        features.append(np.std(zero_crossing_rate))

        # 5. Chroma Özellikleri
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features.append(np.mean(chroma))

        # 6. RMS Enerji
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # 7. Spektral Kontrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features.append(np.mean(contrast))

        return np.array(features)

    except Exception as e:
        print(f"Özellik çıkarılırken hata oluştu: {dosya_yolu} - {e}")
        return None


def test_audio_file(model_path, scaler_path, label_path, audio_filename, model_info_path=None):
    """
    Tek bir ses dosyasını test eder ve sınıflandırma sonucunu döndürür
    """
    # Sabit dizin yolu (sadece ses dosyası için)
    base_directory = r"C:\Users\user\Desktop"

    # Ses dosyası yolunu oluştur
    audio_path = os.path.join(base_directory, audio_filename)

    # Dosya kontrolü
    if not os.path.exists(audio_path):
        print(f"Hata: Ses dosyası bulunamadı: {audio_path}")
        return

    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı: {model_path}")
        return

    if not os.path.exists(scaler_path):
        print(f"Hata: Scaler dosyası bulunamadı: {scaler_path}")
        return

    if not os.path.exists(label_path):
        print(f"Hata: Etiket dosyası bulunamadı: {label_path}")
        return

    # Model bilgilerini yükle (varsa)
    model_tipi = 'dnn'  # Varsayılan model tipi
    max_uzunluk = 5  # Varsayılan maksimum ses uzunluğu

    if model_info_path and os.path.exists(model_info_path):
        try:
            with open(model_info_path, 'r') as f:
                model_bilgileri = json.load(f)
                model_tipi = model_bilgileri.get('model_tipi', model_tipi)
                max_uzunluk = model_bilgileri.get('max_uzunluk', max_uzunluk)
                print(f"Model tipi: {model_tipi}, Maksimum ses uzunluğu: {max_uzunluk} saniye")
        except Exception as e:
            print(f"Model bilgileri yüklenemedi: {e}")

    # Etiket isimlerini yükle
    with open(label_path, 'r') as f:
        etiket_isimleri = json.load(f)
    print(f"Yüklenen etiket sınıfları: {', '.join(etiket_isimleri)}")

    # Modeli yükle
    print(f"Model yükleniyor: {model_path}")
    model = load_model(model_path)

    # Scaler'ı yükle
    print(f"Özellik ölçekleyici yükleniyor: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Ses dosyasından özellikleri çıkar
    print(f"Ses özelliklerini çıkarma: {audio_path}")
    features = extract_features(audio_path, hedef_uzunluk=max_uzunluk)

    if features is None:
        print("Özellik çıkarma başarısız oldu.")
        return

    # Özellikleri ölçeklendir
    features_scaled = scaler.transform(features.reshape(1, -1))

    # CNN modeli için şekil değiştirme
    if model_tipi == 'cnn':
        features_scaled = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
        print("CNN modeli için özellikler şekillendirildi.")

    # Tahmin yap
    print("Tahmini gerçekleştirme...")
    prediction_probs = model.predict(features_scaled, verbose=0)[0]
    predicted_class_index = np.argmax(prediction_probs)
    predicted_class = etiket_isimleri[predicted_class_index]

    # Sonuçları göster
    print("\n" + "=" * 50)
    print("TAHMİN SONUÇLARI")
    print("=" * 50)
    print(f"Dosya: {audio_path}")
    print(f"Tahmin edilen sınıf: {predicted_class}")
    print(
        f"Güven düzeyi: {prediction_probs[predicted_class_index]:.4f} (%{prediction_probs[predicted_class_index] * 100:.2f})")
    print("\nTüm sınıf olasılıkları:")

    # Sınıfları olasılıklarına göre sırala ve yazdır
    for i, (sinif, olasilik) in enumerate(sorted(
            [(etiket_isimleri[i], float(prediction_probs[i])) for i in range(len(etiket_isimleri))],
            key=lambda x: x[1],
            reverse=True
    )):
        print(f"{i + 1}. {sinif}: {olasilik:.4f} (%{olasilik * 100:.2f})")


def main():
    # Varsayılan dosya yolları - kendi dosya yollarınızla değiştirin
    model_path = "ses_siniflandirma_dnn_modeli.h5"  # veya ses_siniflandirma_cnn_modeli.h5
    scaler_path = "scaler.joblib"
    label_path = "etiket_isimleri.json"
    model_info_path = "model_bilgileri.json"

    # Kullanıcıdan sadece dosya adını al
    audio_filename = input("Test edilecek ses dosyasının adını girin (örn: test.wav): ")

    # Testi gerçekleştir
    test_audio_file(model_path, scaler_path, label_path, audio_filename, model_info_path)


if __name__ == "__main__":
    main()