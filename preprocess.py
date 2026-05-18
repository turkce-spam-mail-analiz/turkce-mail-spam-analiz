import re
import snowballstemmer

stemmer = snowballstemmer.stemmer("turkish")

STOP_WORDS = {
    "ve", "veya", "ile", "bir", "bu", "şu", "o", "de", "da",
    "mi", "mı", "mu", "mü", "için", "gibi", "çok", "az",
    "daha", "en", "olarak", "olan", "olduğu", "ise", "ama",
    "fakat", "ancak", "çünkü", "ki", "ne", "ya", "hem",
    "her", "tüm", "kadar", "sonra", "önce", "üzere", "tarafından"
}


def temizle_metin(metin):
    metin = str(metin).lower()

    # URL temizleme
    metin = re.sub(r"http\S+|www\S+", " ", metin)

    # Sayıları temizleme
    metin = re.sub(r"\d+", " ", metin)

    # Noktalama, emoji ve özel karakter temizleme
    metin = re.sub(r"[^a-zçğıöşü\s]", " ", metin)

    # Gereksiz boşluk temizleme
    metin = re.sub(r"\s+", " ", metin).strip()

    # Tokenization
    kelimeler = metin.split()

    # Stop words temizleme
    kelimeler = [kelime for kelime in kelimeler if kelime not in STOP_WORDS]

    # Kök/gövde bulma
    kelimeler = stemmer.stemWords(kelimeler)

    return " ".join(kelimeler)