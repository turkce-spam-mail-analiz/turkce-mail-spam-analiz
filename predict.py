import joblib

model = joblib.load("model/spam_model.pkl")

while True:
    mail = input("\nMail gir (çıkmak için q): ")

    if mail.lower() == "q":
        break

    tahmin = model.predict([mail])[0]
    olasilik = model.predict_proba([mail]).max()

    print("\nSonuç:", tahmin.upper())
    print("Güven:", round(olasilik * 100, 2), "%")