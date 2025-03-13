import numpy as np
import tensorflow as tf
import re
import string
from nltk.corpus import stopwords
import nltk
import asyncio
import logging
import joblib
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv
import os

MAX_VOCAB_SIZE = 25000
MAX_SEQUENCE_LENGTH = 175

load_dotenv()

TOKEN = os.getenv("API_TOKEN")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher()

scaler = joblib.load("scaler.pkl")

model = tf.keras.models.load_model("bilstm_reviewguard.h5")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

with open("tokenizer.json", "r") as f:
    tokenizer_data = f.read()
tokenizer = tokenizer_from_json(tokenizer_data)

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Удаление пунктуации
    text = re.sub(r"\\d+", "", text)  # Удаление чисел
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Удаление стоп-слов
    return " ".join(words)

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    return padded_sequence


# === Функция для генерации дополнительных признаков ===
def extract_features(text):
    review_length = len(text.split())  # Количество слов
    avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0  # Средняя длина слова
    num_uppercase = sum(1 for c in text if c.isupper())  # Количество заглавных букв

    features = np.array([[review_length, avg_word_length, num_uppercase]])
    return scaler.transform(features)

# Обработчик команды /start
@dp.message(Command("start"))
async def start_handler(message: Message):
    await message.answer(        "👋 Привет! Я бот для определения фейковых отзывов.\n\n"
        "🔍 Отправьте мне текст отзыва, и я скажу, настоящий он или поддельный.\n\n"
        "📌 Просто напишите отзыв в сообщении, и я сразу его проанализирую!"
    )

@dp.message(Command("help"))
async def send_help(message: Message):
    help_text = (
        "🆘 **Как пользоваться ботом?**\n\n"
        "🔹 Просто отправьте мне текст отзыва, и я определю, настоящий он или фейковый.\n"
        "🔹 Если бот не отвечает или возникли ошибки, попробуйте перезапустить его.\n\n"
        "Если у вас есть вопросы, напишите разработчику. 😉"
    )
    await message.answer(help_text, parse_mode="Markdown")

# Предсказание с дополнительными признаками
@dp.message()
async def classify_review(message: Message):
    text = message.text

    text_fresh = clean_text(text)
    text_input = preprocess_text(text_fresh)
    features_input = extract_features(text_fresh)

    prediction = model.predict([text_input, features_input])[0][0]

    sentiment = "❌ Фейковый отзыв" if prediction < 0.5 else "✅ Настоящий отзыв"

    await message.answer(sentiment)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
