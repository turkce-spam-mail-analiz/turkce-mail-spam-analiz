import streamlit as st
import joblib

from preprocess import temizle_metin

model = joblib.load("model/spam_model.pkl")

st.set_page_config(page_title="Türkçe Mail Spam Analizi")

st.title("📧 Türkçe Mail Spam Analizi")

mail = st.text_area("Mail metnini girin")

if st.button("Analiz Et"):

    if mail.strip():

        temiz_mail = temizle_metin(mail)
        tahmin = model.predict([temiz_mail])[0]
        olasilik = model.predict_proba([temiz_mail]).max()

        st.subheader("Sonuç")

        if tahmin == "spam":
            st.error(f"SPAM (%{round(olasilik*100,2)})")
        else:
            st.success(f"NORMAL (%{round(olasilik*100,2)})")

    else:
        st.warning("Lütfen mail girin.")