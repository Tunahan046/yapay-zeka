import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df_train = pd.read_csv('yorum_veriseti.csv', encoding='utf-8')
df_test = pd.read_csv('yorumlar.csv', encoding='utf-8', usecols=['Yorum'])

df_train.dropna(subset=['text', 'label'], inplace=True)
df_test.dropna(subset=['Yorum'], inplace=True)

stopwords_tr = set(stopwords.words('turkish'))


def pre_process(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub("[^abcçdefgğhıijklmnoöprsştuüvyz ]", "", text)

    text_tokens = word_tokenize(text)

    text_tokens = [word for word in text_tokens if word not in stopwords_tr]

    lemma = WordNetLemmatizer()
    text_tokens = [lemma.lemmatize(word) for word in text_tokens]

    # Return cleaned text
    return " ".join(text_tokens)


df_train['clean_text'] = df_train['text'].apply(pre_process)
df_test['clean_text'] = df_test['Yorum'].apply(pre_process)

df_train = df_train[df_train['clean_text'] != ""]
df_test = df_test[df_test['clean_text'] != ""]

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df_train['clean_text'])

X_train = tokenizer.texts_to_sequences(df_train['clean_text'])
X_test = tokenizer.texts_to_sequences(df_test['clean_text'])

max_length = 100
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

y_train = df_train['label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_train = to_categorical(y_train)


model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_length))
model.add(SpatialDropout1D(0.2))

model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)))

model.add(Dense(64, activation='relu'))

num_classes = y_train.shape[1]
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stop])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

y_pred = np.argmax(model.predict(X_test), axis=1)

sample_predictions = model.predict(X_test[:10])
for i, pred in enumerate(sample_predictions):
    predicted_class = np.argmax(pred)
    class_label = label_encoder.inverse_transform([predicted_class])[0]
    print(f"Sample {i + 1}: {class_label} (Confidence: {pred[predicted_class]:.2f})")
model.save('model.h5')