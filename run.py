import os
import sys
from app import create_app

# Set console encoding to UTF-8 to handle Unicode characters
if sys.platform.startswith('win'):
    # For Windows, set console to handle UTF-8
    os.system('chcp 65001 > nul')
    # Set PYTHONIOENCODING to UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'

app = create_app()

if __name__ == '__main__':
    app.run(
        host=app.config['BACKEND_URL'],
        port=app.config['BACKEND_PORT'],
        debug=app.config['DEBUG'],
        threaded=True
    )