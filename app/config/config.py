import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    DB_CONFIG = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }
    BACKEND_URL = os.getenv('BACKEND_URL', 'localhost')
    BACKEND_PORT = int(os.getenv('BACKEND_PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'DEV')
    MODEL_API_KEY = os.getenv('MODEL_API_KEY', '')
    
    # API Keys for new models
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', '')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')
    
    # Poetry Search Engine Configuration
    POETRY_MODEL_ENABLED = os.getenv('POETRY_MODEL_ENABLED', 'True').lower() in ('true', '1', 't')
    POETRY_CACHE_SIZE = int(os.getenv('POETRY_CACHE_SIZE', 100))
    POETRY_CACHE_EXPIRY_HOURS = int(os.getenv('POETRY_CACHE_EXPIRY_HOURS', 24))
    POETRY_DATA_PATH = os.getenv('POETRY_DATA_PATH', 'app/models/poetry/data/poetry_data.csv')
    
    # General Chatbot Configuration
    CHATBOT_MODEL_ENABLED = os.getenv('CHATBOT_MODEL_ENABLED', 'True').lower() in ('true', '1', 't')
    CHATBOT_MODEL_NAME = os.getenv('CHATBOT_MODEL_NAME', 'BAAI/bge-m3')
    CHATBOT_DATA_PATH = os.getenv('CHATBOT_DATA_PATH', 'app/models/general_chatbot/data/bhit_data.txt')
    CHATBOT_CACHE_SIZE = int(os.getenv('CHATBOT_CACHE_SIZE', 2000))
    CHATBOT_MAX_CONVERSATION_HISTORY = int(os.getenv('CHATBOT_MAX_CONVERSATION_HISTORY', 10))
    CHATBOT_INITIAL_RETRIEVE_K = int(os.getenv('CHATBOT_INITIAL_RETRIEVE_K', 30))
    CHATBOT_RERANK_K = int(os.getenv('CHATBOT_RERANK_K', 15))
    CHATBOT_FINAL_CONTEXT_K = int(os.getenv('CHATBOT_FINAL_CONTEXT_K', 6))
    CHATBOT_MIN_SIMILARITY_THRESHOLD = float(os.getenv('CHATBOT_MIN_SIMILARITY_THRESHOLD', 0.1))
    CHATBOT_CONTEXT_EXPANSION_RADIUS = int(os.getenv('CHATBOT_CONTEXT_EXPANSION_RADIUS', 2))
    
    