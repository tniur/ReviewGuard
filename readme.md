# ReviewGuard Bot

Telegram-бот для классификации отзывов как настоящих или фейковых.

## Требования

- Python 3.8+
- Установите зависимости из `requirements.txt`

## Установка

1. Клонируйте репозиторий:

    ```bash
    git clone https://github.com/yourusername/fake-review-classifier-bot.git
    cd fake-review-classifier-bot
    ```

2. Создайте виртуальное окружение и активируйте его:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Для Linux/macOS
    venv\Scripts\activate  # Для Windows
    ```

3. Установите зависимости:

    ```bash
    pip install -r requirements.txt
    ```

4. Создайте файл `.env` в корне проекта и добавьте ваш API_TOKEN:

    ```
    API_TOKEN=your_telegram_bot_token
    ```

## Запуск

1. Запустите бота:

    ```bash
    python main.py
    ```