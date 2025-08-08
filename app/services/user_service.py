from flask import g
from ..utils.database import get_db

def get_user_details():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s AND status = TRUE", (g.user_id,))
    user = cursor.fetchone()
    
    if not user:
        return {'error': 'User not found'}, 404
    
    return {'user': user}, 200