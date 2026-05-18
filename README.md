# 📧 Türkçe Mail Spam Analizi

Bu proje, Türkçe e-posta içeriklerini spam veya normal olarak sınıflandırabilen bir Doğal Dil İşleme (NLP) ve Makine Öğrenmesi uygulamasıdır. Proje kapsamında manuel olarak oluşturulmuş Türkçe e-posta veri seti kullanılmış ve TF-IDF + Logistic Regression yöntemleri ile bir spam tespit sistemi geliştirilmiştir.

---

# 🚀 Proje Amacı

Bu çalışmanın amacı, Türkçe e-posta içeriklerini analiz ederek spam içerikleri otomatik şekilde tespit edebilen bir sistem geliştirmektir.

Proje kapsamında:

- Türkçe e-posta veri seti oluşturulmuştur
- Veri ön işleme işlemleri uygulanmıştır
- Makine öğrenmesi modeli eğitilmiştir
- Model başarı analizi gerçekleştirilmiştir
- Web tabanlı kullanıcı arayüzü geliştirilmiştir

---

# 📊 Veri Seti

Veri seti manuel olarak hazırlanmıştır.

## Veri Özellikleri

| Özellik | Değer |
|---|---|
| Başlangıç veri sayısı | 12.421 |
| Temizlik sonrası veri | 8.342 |
| Spam veri sayısı | 4.353 |
| Normal veri sayısı | 3.989 |
| Veri dili | Türkçe |
| Dosya formatı | UTF-8 CSV |

---

## Kullanılan Kategoriler

- Banka
- Kargo
- Sosyal Medya
- Alışveriş
- Teknoloji
- Günlük İletişim
- Yatırım
- Kampanya

---

# 🧹 Veri Ön İşleme Adımları

Projede aşağıdaki NLP ön işleme işlemleri uygulanmıştır:

- Küçük harfe dönüştürme
- Noktalama işaretlerini temizleme
- Sayı temizleme
- URL temizleme
- Emoji temizleme
- Gereksiz boşluk temizleme
- Stop words temizleme
- Tokenization
- Kök bulma (Stemming)

---

# 🧠 Kullanılan Teknolojiler

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Matplotlib
- Seaborn
- SnowballStemmer

---

# 🤖 Model Bilgileri

Projede metin sınıflandırma işlemi için TF-IDF + Logistic Regression yöntemi kullanılmıştır.

## Eğitim/Test Ayrımı

- Eğitim Verisi: %80
- Test Verisi: %20

---

# 📈 Model Sonuçları

| Metrik | Sonuç |
|---|---|
| Accuracy | %96 |
| Precision | %96 |
| Recall | %96 |
| F1-Score | %96 |

Model spam ve normal e-postaları yüksek doğruluk oranı ile sınıflandırabilmektedir.

---

# 📊 Confusion Matrix

Projede model performansını değerlendirmek amacıyla confusion matrix kullanılmıştır.

## Örnek Sonuç

| Gerçek / Tahmin | Normal | Spam |
|---|---|---|
| Normal | 767 | 40 |
| Spam | 24 | 839 |

---

# 🌐 Streamlit Arayüzü

Proje kapsamında kullanıcıların e-posta metinlerini analiz edebilmesi için Streamlit tabanlı web arayüzü geliştirilmiştir.

## Kullanıcı İş Akışı

1. Mail metni girilir
2. Sistem analizi gerçekleştirir
3. Spam / Normal sonucu gösterilir

---

# 📂 Proje Yapısı

```bash
turkce-mail-spam-analizi/
│
├── data/
│   └── mailler.csv
│
├── model/
│   └── spam_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── veri_dagilimi.png
│   ├── spam_en_sik_kelimeler.png
│   ├── normal_en_sik_kelimeler.png
│   └── model_sonuclari.txt
│
├── preprocess.py
├── train.py
├── app.py
└── README.md
