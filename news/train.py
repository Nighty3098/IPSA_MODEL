# train.py

import pickle
import re

import numpy as np
import pandas as pd
from data import data
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

# Создание DataFrame
df = pd.DataFrame(data)


# Очистка текста
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Удаляем все символы кроме букв
    return text.lower()


df["cleaned_text"] = df["text"].apply(clean_text)

# Токенизация текста
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["cleaned_text"])
sequences = tokenizer.texts_to_sequences(df["cleaned_text"])
X = pad_sequences(sequences, maxlen=100)

# Метки
y = np.array(df["label"])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создание модели
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))  # Бинарная классификация

# Компиляция модели
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Сохранение модели и токенизатора
model.save("sentiment_model.h5")
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle)
