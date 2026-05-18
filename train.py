import pandas as pd
import re
import joblib
import os

from preprocess import temizle_metin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# CSV dosyasını oku
df = pd.read_csv(
    "data/mailler.csv",
    sep=";",
    encoding="utf-8",
    on_bad_lines="skip"
)

print("İlk sütunlar:", df.columns)
print("Başlangıç veri sayısı:", len(df))

# Sütun adlarını standartlaştır
df.columns = df.columns.str.lower().str.strip()

# Gerekli sütun kontrolü
gerekli_sutunlar = ["metin", "etiket"]

for sutun in gerekli_sutunlar:
    if sutun not in df.columns:
        raise ValueError(f"'{sutun}' sütunu bulunamadı. CSV dosyanızdaki sütun adlarını kontrol edin.")

# Boş metin ve boş etiketleri sil
df = df.dropna(subset=["metin", "etiket"])

# Etiketleri standart hale getir
df["etiket"] = df["etiket"].astype(str).str.lower().str.strip()

# Sadece spam ve normal etiketlerini al
df = df[df["etiket"].isin(["spam", "normal"])]

# Metinleri temizle
df["metin_temiz"] = df["metin"].apply(temizle_metin)

# Temizlik sonrası boş kalan metinleri sil
df = df[df["metin_temiz"].str.len() > 0]

# Çok kısa metinleri çıkar
df = df[df["metin_temiz"].str.split().str.len() >= 4]

# Tekrarlı metinleri sil
df = df.drop_duplicates(subset=["metin_temiz"])

print("Temizlik sonrası veri sayısı:", len(df))
print("\nEtiket dağılımı:")
print(df["etiket"].value_counts())

# Temizlenmiş veri setini kaydet
df.to_csv("outputs/temizlenmis_veri_seti.csv", index=False, encoding="utf-8-sig")

X = df["metin_temiz"]
y = df["etiket"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nEğitim veri sayısı:", len(X_train))
print("Test veri sayısı:", len(X_test))

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(matrix)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 5))

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Spam"],
    yticklabels=["Normal", "Spam"]
)

plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("Confusion matrix görseli kaydedildi.")

# Sonuçları txt dosyasına kaydet
with open("outputs/model_sonuclari.txt", "w", encoding="utf-8") as f:
    f.write("Türkçe Mail Spam Analizi Model Sonuçları\n")
    f.write("=======================================\n\n")
    f.write(f"Başlangıç veri sayısı: {len(pd.read_csv('data/mailler.csv', sep=';', encoding='utf-8', on_bad_lines='skip'))}\n")
    f.write(f"Temizlik sonrası veri sayısı: {len(df)}\n\n")
    f.write("Etiket dağılımı:\n")
    f.write(str(df["etiket"].value_counts()))
    f.write("\n\n")
    f.write(f"Eğitim veri sayısı: {len(X_train)}\n")
    f.write(f"Test veri sayısı: {len(X_test)}\n\n")
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))

# Modeli kaydet
joblib.dump(model, "model/spam_model.pkl")

print("\nModel kaydedildi: model/spam_model.pkl")
print("Temizlenmiş veri seti kaydedildi: outputs/temizlenmis_veri_seti.csv")
print("Model sonuçları kaydedildi: outputs/model_sonuclari.txt")

import matplotlib.pyplot as plt

etiket_sayilari = df["etiket"].value_counts()

plt.figure(figsize=(6, 5))

etiket_sayilari.plot(
    kind="bar",
    color=["#4CAF50", "#F44336"]
)

plt.title("Spam ve Normal Veri Dağılımı")
plt.xlabel("Etiket")
plt.ylabel("Veri Sayısı")
plt.xticks(rotation=0)

plt.savefig("outputs/veri_dagilimi.png")
plt.close()

print("Veri dağılım grafiği kaydedildi.")

import matplotlib.pyplot as plt

# Spam metinleri birleştir
spam_metinleri = " ".join(df[df["etiket"] == "spam"]["metin_temiz"])

# Normal metinleri birleştir
normal_metinleri = " ".join(df[df["etiket"] == "normal"]["metin_temiz"])

from collections import Counter
import matplotlib.pyplot as plt

def en_sik_kelimeler_grafigi(metinler, baslik, dosya_adi):
    tum_kelimeler = " ".join(metinler).split()
    kelime_sayilari = Counter(tum_kelimeler).most_common(15)

    kelimeler = [x[0] for x in kelime_sayilari]
    sayilar = [x[1] for x in kelime_sayilari]

    plt.figure(figsize=(10, 6))
    plt.bar(kelimeler, sayilar)
    plt.title(baslik)
    plt.xlabel("Kelimeler")
    plt.ylabel("Frekans")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(dosya_adi)
    plt.close()


en_sik_kelimeler_grafigi(
    df[df["etiket"] == "spam"]["metin_temiz"],
    "Spam Maillerde En Sık Geçen Kelimeler",
    "outputs/spam_en_sik_kelimeler.png"
)

en_sik_kelimeler_grafigi(
    df[df["etiket"] == "normal"]["metin_temiz"],
    "Normal Maillerde En Sık Geçen Kelimeler",
    "outputs/normal_en_sik_kelimeler.png"
)

print("En sık kelime grafikleri kaydedildi.")

plt.figure(figsize=(10, 5))
plt.imshow(normal_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Normal Maillerde En Sık Geçen Kelimeler")
plt.savefig("outputs/normal_wordcloud.png")
plt.close()

print("WordCloud görselleri kaydedildi.")