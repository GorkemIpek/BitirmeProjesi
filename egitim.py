import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import random
import json
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Uyarıları kontrol et
warnings.filterwarnings('ignore')

# Rastgeleliği sabitleyerek tekrarlanabilir sonuçlar elde et
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# GPU belleğini yönet
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU bellek ayarı yapılamadı: {e}")


def extract_features(dosya_yolu, hedef_uzunluk=None, n_mfcc=40, n_fft=2048, hop_length=512):
    """
    Ses dosyasından gelişmiş özellikler çıkarır
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
                # Rastgele bir kesit al
                baslangic = random.uniform(0, orijinal_uzunluk - hedef_uzunluk)
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


def augment_audio(y, sr, augment_methods=None):
    """
    Ses verisine veri artırma (data augmentation) teknikleri uygular
    """
    if augment_methods is None:
        augment_methods = ['pitch', 'speed', 'noise']

    # Orijinal sesi kopyala
    augmented_y = y.copy()

    if 'pitch' in augment_methods:
        # Perde değişimi (pitch shift)
        n_steps = np.random.uniform(-3, 3)
        augmented_y = librosa.effects.pitch_shift(augmented_y, sr=sr, n_steps=n_steps)

    if 'speed' in augment_methods:
        # Hız değişimi (time stretch)
        speed_factor = np.random.uniform(0.9, 1.1)
        augmented_y = librosa.effects.time_stretch(augmented_y, rate=speed_factor)

    if 'noise' in augment_methods:
        # Gürültü ekleme
        noise_factor = np.random.uniform(0.001, 0.01)
        noise = np.random.randn(len(augmented_y))
        augmented_y = augmented_y + noise_factor * noise

    return augmented_y


def load_and_preprocess_data(ana_dizin, max_uzunluk=None, augment=True, n_augmentation=1):
    """
    Veri setini yükler, ön işler ve veri artırma uygular
    """
    ozellikler = []
    etiketler = []
    dosya_yollari = []  # Dosya yollarını takip et

    klasorler = ["Silah", "Korna", "Scream"]
    sinif_sayaclari = {k: 0 for k in klasorler}  # Her sınıftaki örnek sayısını takip et

    for etiket in klasorler:
        klasor_yolu = os.path.join(ana_dizin, etiket)
        if os.path.exists(klasor_yolu):
            dosya_listesi = [f for f in os.listdir(klasor_yolu) if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
            print(f"{etiket} klasöründe {len(dosya_listesi)} ses dosyası bulundu.")

            for dosya_adi in dosya_listesi:
                dosya_yolu = os.path.join(klasor_yolu, dosya_adi)

                # Orijinal ses özelliklerini çıkar
                features = extract_features(dosya_yolu, hedef_uzunluk=max_uzunluk)

                if features is not None:
                    ozellikler.append(features)
                    etiketler.append(etiket)
                    dosya_yollari.append(dosya_yolu)
                    sinif_sayaclari[etiket] += 1

                    # Veri artırma uygula
                    if augment:
                        try:
                            y, sr = librosa.load(dosya_yolu, sr=None)
                            for _ in range(n_augmentation):
                                augmented_y = augment_audio(y, sr)

                                # Artırılmış ses verisinden özellik çıkar
                                aug_features = extract_features_from_array(augmented_y, sr, hedef_uzunluk=max_uzunluk)

                                if aug_features is not None:
                                    ozellikler.append(aug_features)
                                    etiketler.append(etiket)
                                    dosya_yollari.append(f"{dosya_yolu} (artırılmış)")
                                    sinif_sayaclari[etiket] += 1
                        except Exception as e:
                            print(f"Veri artırma sırasında hata: {dosya_yolu} - {e}")
        else:
            print(f"Uyarı: Klasör bulunamadı: {klasor_yolu}")

    # Sınıf dağılımını yazdır
    print("\nSınıf dağılımı:")
    for sinif, sayi in sinif_sayaclari.items():
        print(f"{sinif}: {sayi} örnek")

    if not ozellikler:
        raise ValueError("Hiç geçerli özellik çıkarılamadı! Veri yolunu ve dosyaları kontrol edin.")

    return np.array(ozellikler), np.array(etiketler), dosya_yollari


def extract_features_from_array(y, sr, hedef_uzunluk=None, n_mfcc=40, n_fft=2048, hop_length=512):
    """
    Ses dizisinden (array) özellik çıkarma (veri artırma için)
    """
    try:
        # Hedef uzunluğa göre sesi kesme veya padding yapma
        orijinal_uzunluk = len(y) / sr
        if hedef_uzunluk is not None:
            if orijinal_uzunluk > hedef_uzunluk:
                # Rastgele bir kesit al
                baslangic = random.uniform(0, orijinal_uzunluk - hedef_uzunluk)
                y = y[int(baslangic * sr):int((baslangic + hedef_uzunluk) * sr)]
            elif orijinal_uzunluk < hedef_uzunluk:
                # Kısa ses dosyalarını 0 ile doldur
                padding = np.zeros(int(hedef_uzunluk * sr) - len(y))
                y = np.concatenate((y, padding))

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
        print(f"Artırılmış veriden özellik çıkarılırken hata oluştu: {e}")
        return None


def create_model(input_shape, num_classes, model_type='cnn'):
    """
    Farklı model mimarileri oluşturma
    """
    if model_type == 'dnn':
        # DNN Modeli (Geliştirilmiş)
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

    elif model_type == 'cnn':
        # 1D CNN Modeli
        inputs = Input(shape=(input_shape, 1))
        x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)

    else:
        raise ValueError(f"Bilinmeyen model tipi: {model_type}")

    return model


def plot_training_history(history, model_name='model'):
    """
    Eğitim geçmişini görselleştirme
    """
    # Eğitim geçmişini görselleştirme
    plt.figure(figsize=(12, 4))

    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim')
    plt.plot(history.history['val_accuracy'], label='Doğrulama')
    plt.title('Model Doğruluğu')
    plt.xlabel('Epok')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim')
    plt.plot(history.history['val_loss'], label='Doğrulama')
    plt.title('Model Kaybı')
    plt.xlabel('Epok')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{model_name}_training_history.png")
    plt.close()
    print(f"Eğitim geçmişi kaydedildi: {model_name}_training_history.png")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name='model'):
    """
    Karmaşıklık matrisini görselleştirme
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()
    print(f"Karmaşıklık matrisi kaydedildi: {model_name}_confusion_matrix.png")


def main():
    # Parametreler
    ana_dizin = r"C:\Users\user\Desktop\BitirmeDataset"
    maksimum_uzunluk = 5  # 5 saniyelik sabit uzunluk
    augment = True  # Veri artırma kullan
    n_augmentation = 2  # Her ses dosyası için 2 artırılmış örnek oluştur
    model_type = 'dnn'  # 'dnn' veya 'cnn'
    epochs = 100  # Maksimum epok sayısı
    batch_size = 32
    learning_rate = 0.001
    patience = 15  # Erken durdurma sabır değeri

    print(f"Ses verileri yükleniyor: {ana_dizin}")
    try:
        # Veri yükleme ve ön işleme
        ozellikler, etiketler, dosya_yollari = load_and_preprocess_data(
            ana_dizin,
            max_uzunluk=maksimum_uzunluk,
            augment=augment,
            n_augmentation=n_augmentation
        )

        # Özellik ölçeklendirme (standardizasyon)
        scaler = StandardScaler()
        ozellikler_scaled = scaler.fit_transform(ozellikler)

        # Etiketleri sayısal değerlere dönüştürme
        label_encoder = LabelEncoder()
        etiketler_encoded = label_encoder.fit_transform(etiketler)

        # Etiket isimlerini kaydetme (test için)
        etiket_isimleri = label_encoder.classes_.tolist()
        with open("etiket_isimleri.json", "w") as f:
            json.dump(etiket_isimleri, f)
        print("Etiket isimleri kaydedildi: etiket_isimleri.json")

        # Scaler'ı kaydet (test için)
        import joblib
        joblib.dump(scaler, 'scaler.joblib')
        print("Özellik ölçekleyici kaydedildi: scaler.joblib")

        # Kategorik etiketlere dönüştürme (one-hot encoding)
        num_classes = len(np.unique(etiketler_encoded))
        y_categorical = to_categorical(etiketler_encoded, num_classes=num_classes)

        print(f"Toplam örnek sayısı: {len(ozellikler_scaled)}")
        print(f"Özellik boyutu: {ozellikler_scaled.shape[1]}")
        print(f"Sınıf sayısı: {num_classes}")

        # Veri setini eğitim ve test kümelerine ayırma
        X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
            ozellikler_scaled,
            y_categorical,
            dosya_yollari,
            test_size=0.2,
            random_state=42,
            stratify=etiketler_encoded
        )

        print(f"Eğitim veri boyutu: {X_train.shape}")
        print(f"Test veri boyutu: {X_test.shape}")

        # CNN için veri şeklini ayarla
        if model_type == 'cnn':
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            input_shape = X_train.shape[1]
        else:
            input_shape = X_train.shape[1]

        # Model oluşturma
        model = create_model(input_shape, num_classes, model_type=model_type)

        # Modeli derleme
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Model özeti
        model.summary()

        # Callback'ler
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(f"best_{model_type}_model.h5", monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

        # Modeli eğitme
        print(f"\nModel eğitimi başlıyor ({model_type.upper()})...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        # Eğitim geçmişini görselleştirme
        plot_training_history(history, model_name=model_type)

        # Modeli değerlendirme
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"\nTest doğruluğu: {accuracy:.4f}")
        print(f"Test kaybı: {loss:.4f}")

        # Tahminler
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Karmaşıklık matrisi
        plot_confusion_matrix(y_true, y_pred, etiket_isimleri, model_name=model_type)

        # Sınıflandırma raporu
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_true, y_pred, target_names=etiket_isimleri))

        # Test setindeki en kötü tahmin edilen örnekleri bul
        incorrect_indices = np.where(y_pred != y_true)[0]
        if len(incorrect_indices) > 0:
            print("\nYanlış sınıflandırılan bazı örnekler:")
            for i in range(min(5, len(incorrect_indices))):
                idx = incorrect_indices[i]
                print(f"Dosya: {test_files[idx]}")
                print(f"Gerçek etiket: {etiket_isimleri[y_true[idx]]}")
                print(f"Tahmin edilen: {etiket_isimleri[y_pred[idx]]}")
                print(
                    f"Tahmin olasılıkları: {', '.join([f'{etiket_isimleri[j]}: {y_pred_probs[idx][j]:.2f}' for j in range(num_classes)])}")
                print()

        # Modeli kaydetme
        model_kayit_adi = f"ses_siniflandirma_{model_type}_modeli.h5"
        model.save(model_kayit_adi)
        print(f"Model kaydedildi: {model_kayit_adi}")

        # Model bilgilerini kaydetme
        model_bilgileri = {
            "model_tipi": model_type,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "sinif_isimleri": etiket_isimleri,
            "test_accuracy": float(accuracy),
            "test_loss": float(loss),
            "feature_count": ozellikler_scaled.shape[1],
            "max_uzunluk": maksimum_uzunluk
        }

        with open("model_bilgileri.json", "w") as f:
            json.dump(model_bilgileri, f)
        print("Model bilgileri kaydedildi: model_bilgileri.json")

        print("\nTüm işlemler başarıyla tamamlandı!")

    except Exception as e:
        print(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()