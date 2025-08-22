import logging
import os
import uuid
from logging.handlers import RotatingFileHandler
from config import Settings

def setup_loggers():
    os.makedirs(Settings.LOGS_DIR, exist_ok=True)
    os.makedirs(Settings.CONVERSATION_LOGS_DIR, exist_ok=True)

    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    app_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    app_handler = RotatingFileHandler(os.path.join(Settings.LOGS_DIR, "app.log"), maxBytes=5*1024*1024, backupCount=2)
    app_handler.setFormatter(app_formatter)
    if not app_logger.handlers:
        app_logger.addHandler(app_handler)

    cost_logger = logging.getLogger("cost")
    cost_logger.setLevel(logging.INFO)
    cost_formatter = logging.Formatter('%(asctime)s - %(message)s')
    cost_handler = logging.FileHandler(os.path.join(Settings.LOGS_DIR, "cost_usage.log"))
    cost_handler.setFormatter(cost_formatter)
    if not cost_logger.handlers:
        cost_logger.addHandler(cost_handler)

    rag_logger = logging.getLogger("rag")
    rag_logger.setLevel(logging.DEBUG)
    rag_formatter = logging.Formatter('%(asctime)s - CONV_ID: %(conv_id)s - %(message)s')
    rag_handler = logging.FileHandler(os.path.join(Settings.LOGS_DIR, "rag_details.log"))
    rag_handler.setFormatter(rag_formatter)
    if not rag_logger.handlers:
        rag_logger.addHandler(rag_handler)

    return app_logger, cost_logger, rag_logger

def log_conversation(conversation_id: uuid.UUID, user_query: str, bot_response: str):
    log_path = os.path.join(Settings.CONVERSATION_LOGS_DIR, f"{conversation_id}.log")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"--- Conversation ID: {conversation_id} ---\n")
        f.write(f"Timestamp: {logging.Formatter().formatTime(logging.makeLogRecord({}))}\n")
        f.write("="*50 + "\n")
        f.write(f"User Query:\n{user_query}\n")
        f.write("="*50 + "\n")
        f.write(f"Bot Response:\n{bot_response}\n")
        f.write("="*50 + "\n")

app_logger, cost_logger, rag_logger = setup_loggers()