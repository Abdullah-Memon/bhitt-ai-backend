import uuid
import jwt
import bcrypt
from datetime import datetime, timedelta
from flask import current_app, g
from ..utils.database import get_db
from .subscription_service import ensure_basic_subscription

def login_user(data):
    try:
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return {'error': 'اي ميل ۽ پاس ورڊ لازمي آهن'}, 400
        
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s AND status = TRUE", (email,))
        user = cursor.fetchone()
        
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return {'error': 'غلط اي ميل يا پاس ورڊ'}, 401
        
        user.pop('password_hash', None)

        # Get user's latest active subscription via transactions
        cursor.execute("""
            SELECT s.max_queries_per_day FROM transactions t
            JOIN subscriptions s ON t.subscription_id = s.id
            WHERE t.user_id = %s AND s.is_active = TRUE AND t.status = TRUE
            ORDER BY t.created_at DESC LIMIT 1
        """, (user['id'],))
        subscription = cursor.fetchone()
        max_queries = subscription['max_queries_per_day'] if subscription else 100
        
        token = jwt.encode({
            'user_id': user['id'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, current_app.config['SECRET_KEY'])
        
        return {
            'user': user,
            'token': token,
            'max_queries_per_day': max_queries,
            'message': 'ڪاميابي سان لاگ ان ٿيا'
        }, 200
    except Exception as e:
        return {'error': 'لاگ ان ۾ خرابي: نيٽ ورڪ يا سرور جي مسئلي جي ڪري ٻيهر ڪوشش ڪريو'}, 500

def register_user(data):
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if not all([name, email, password]):
        return {'error': 'Missing required fields'}, 400
    
    db = get_db()
    cursor = db.cursor(dictionary=True)
    
    try:
        user_id = str(uuid.uuid4())
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        cursor.execute("""
            INSERT INTO users (id, name, email, password_hash)
            VALUES (%s, %s, %s, %s)
        """, (user_id, name, email, password_hash))
        db.commit()
        
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        # Ensure Basic subscription exists
        ensure_basic_subscription()
        # Get Basic subscription id and max_queries_per_day
        cursor.execute("SELECT id, max_queries_per_day FROM subscriptions WHERE package_name = %s", ('Basic',))
        basic = cursor.fetchone()
        if basic:
            subscription_id = basic['id']
            max_queries = basic['max_queries_per_day']
            transaction_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO transactions (id, user_id, subscription_id, payment_method, amount, payment_status)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (transaction_id, user_id, subscription_id, 'credit_card', 0, 'success'))
            # Set remaining_queries for today
            cursor.execute("UPDATE users SET remaining_queries = %s WHERE id = %s", (max_queries, user_id))
            db.commit()
        return {'user': user}, 201
    except Exception as e:
        return {'error': f'Registration failed: {str(e)}'}, 400

def verify_token(token):
    if not token:
        return {'error': 'Token is missing'}, 401
    
    try:
        token = token.split(" ")[1] if token.startswith("Bearer ") else token
        data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
        return {'valid': True, 'user_id': data['user_id']}, 200
    except:
        return {'valid': False, 'error': 'Invalid token'}, 401