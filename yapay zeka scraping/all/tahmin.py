import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk import word_tokenize, WordNetLemmatizer
import re
import numpy as np

df_comments = pd.read_csv('yorumlar.csv', encoding='utf-8', usecols=['Yorum'])

def pre_process(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub("[^abcçdefgğhıijklmnoöprsştuüvyz ]", "", text)
    text = word_tokenize(text)
    lemma = WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    return " ".join(text)

df_comments['clean_text'] = df_comments['Yorum'].apply(pre_process)

try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer yüklendi.")
except FileNotFoundError:
    print("Tokenizer dosyası bulunamadı. Lütfen tokenizer'ı kaydedin ve tekrar deneyin.")
    exit()

X_comments = tokenizer.texts_to_sequences(df_comments['clean_text'])
max_length = 100
X_comments = pad_sequences(X_comments, maxlen=max_length)

model = load_model('model.h5')

y_pred = model.predict(X_comments)

y_pred_classes = np.argmax(y_pred, axis=1)


try:

    label_encoder = LabelEncoder()

    label_encoder.fit(
        ['negatif', 'pozitif', 'nötr'])


    predicted_classes = label_encoder.inverse_transform(y_pred_classes)

    df_comments['Predicted_Class'] = predicted_classes


    df_comments['Vectorized_Text'] = [' '.join(map(str, seq)) for seq in X_comments]

    df_comments.to_csv('yorumlar_tahmin.csv', index=False, columns=['Yorum', 'Predicted_Class', 'Vectorized_Text'])
    print("Tahminler ve vektörleştirilmiş veriler kaydedildi: yorumlar_tahmin.csv")

except KeyError:
    print("Etiket sütunu (label) bulunamadı. Lütfen eğitim verinizi ve etiket sütununu kontrol edin.")
