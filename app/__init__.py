from flask import Flask
from flask_cors import CORS  # Added for CORS support
from .config.config import Config
from .utils.database import init_db
from .utils.limiter import init_limiter
from .api.auth import auth_bp
from .api.user import user_bp
from .api.subscription import subscription_bp
from .api.new_models import new_models_bp

def create_app():
    import logging
    from logging.handlers import RotatingFileHandler
    import os

    app = Flask(__name__)
    app.config.from_object(Config)

    # Production logging setup
    if not app.debug:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(os.path.join(log_dir, 'app.log'), maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Production logging is enabled.')
    
    # Configure console handler with UTF-8 encoding for development
    if app.debug:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
        console_handler.setFormatter(formatter)
        # Set encoding to handle Unicode characters
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8')
        app.logger.addHandler(console_handler)
        app.logger.setLevel(logging.INFO)

    # Enable CORS for all routes (you can restrict origins if needed)
    # CORS(app)
    # CORS(app, resources={r"/bhitt-ai/api/*": {"origins": ["http://bhitt-ai.sindh.ai:3001", "https://bhitt-ai.sindh.ai"]}})
    CORS(app, resources={r"/bhitt-ai/api/*": {"origins": "*"}})

    # Example for restricting: CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize database within app context
    with app.app_context():
        init_db()
        # Initialize the new models at startup
        try:
            from .models.poetry.search_engine import SindhiPoetrySearchEngine
            from .models.general_chatbot.chatbot import get_rag_system
            
            # Initialize poetry search engine
            poetry_engine = SindhiPoetrySearchEngine()
            app.logger.info("Poetry search engine initialized.")
            
            # Initialize general chatbot (RAG system)
            rag_system = get_rag_system()
            app.logger.info("General chatbot initialized.")
            
        except Exception as e:
            app.logger.error(f"Error initializing models: {str(e)}")
        
        app.logger.info("New models initialized.")

    # Initialize rate limiter
    init_limiter(app)

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/bhitt-ai/api')
    app.register_blueprint(user_bp, url_prefix='/bhitt-ai/api')
    app.register_blueprint(subscription_bp, url_prefix='/bhitt-ai/api')
    app.register_blueprint(new_models_bp, url_prefix='/bhitt-ai/api')
    app.logger.info("Backend and all models are running.")

    return app