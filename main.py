import numpy as np
import torch
import re
import string
from nltk.corpus import stopwords
import nltk
import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from dotenv import load_dotenv
import os

MAX_VOCAB_SIZE = 25000
MAX_SEQUENCE_LENGTH = 175

load_dotenv()

TOKEN = os.getenv("API_TOKEN")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Загрузка предобученной модели ELECTRA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/electra-small-discriminator"
model = ElectraForSequenceClassification.from_pretrained("./saved_model").to(device)
tokenizer = ElectraTokenizer.from_pretrained("./saved_model")


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    return encoding

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
    text_input = preprocess_text(text)

    with torch.no_grad():
        outputs = model(**text_input)
        prediction = torch.softmax(outputs.logits, dim=1)[:, 1].item()

    sentiment = "❌ Фейковый отзыв" if prediction < 0.5 else "✅ Настоящий отзыв"
    await message.answer(sentiment)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
