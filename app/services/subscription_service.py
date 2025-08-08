import uuid
from flask import g
from ..utils.database import get_db

def get_subscription_details():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT s.* FROM subscriptions s
        JOIN transactions t ON t.subscription_id = s.id
        WHERE t.user_id = %s AND s.is_active = TRUE
        ORDER BY t.created_at DESC LIMIT 1
    """, (g.user_id,))
    subscription = cursor.fetchone()
    if not subscription:
        return {'error': 'No active subscription found'}, 404
    return {'subscription': subscription}, 200

def set_subscription(data):
    subscription_id = data.get('subscription_id')
    if not subscription_id:
        return {'error': 'Subscription ID is required'}, 400
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM subscriptions WHERE id = %s AND is_active = TRUE", (subscription_id,))
        subscription = cursor.fetchone()
        if not subscription:
            return {'error': 'Invalid subscription ID'}, 404
        transaction_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO transactions (id, user_id, subscription_id, payment_method, amount, payment_status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (transaction_id, g.user_id, subscription_id, 'credit_card', subscription['max_queries_per_day'] * 10, 'success'))
        db.commit()
        return {'message': 'Subscription set successfully', 'transaction_id': transaction_id}, 200
    except Exception as e:
        return {'error': f'Transaction failed: {str(e)}'}, 400

def ensure_basic_subscription():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM subscriptions WHERE package_name = %s", ('Basic',))
    basic = cursor.fetchone()
    if not basic:
        basic_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO subscriptions (id, package_name, max_queries_per_day, is_active, status)
            VALUES (%s, %s, %s, TRUE, TRUE)
        """, (basic_id, 'Basic', 10))
        db.commit()
    return