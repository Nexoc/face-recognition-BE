import logging
import os
from dotenv import load_dotenv

load_dotenv()  # Загружаем переменные из .env

def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log", mode="a")
        ]
    )
