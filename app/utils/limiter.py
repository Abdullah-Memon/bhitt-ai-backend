from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import jsonify

# Define limiter at module level (without app)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per minute"],
    storage_uri="memory://"
)

def init_limiter(app):
    limiter.init_app(app)
    
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({'error': 'There is a lot of traffic. Please wait or Try Again.'}), 429
    
    return limiter