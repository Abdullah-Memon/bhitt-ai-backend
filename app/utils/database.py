import mysql.connector
from flask import g, current_app
from ..models.schema import get_schema

def get_db():
    if 'db' not in g:
        g.db = mysql.connector.connect(**current_app.config['DB_CONFIG'])
    return g.db

def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    cursor = db.cursor()
    schemas = get_schema()
    
    for table_name, schema in schemas.items():
        cursor.execute(schema)
    db.commit()

def cleanup_old_sessions(days_old=30):
    """
    Clean up old inactive sessions and their messages
    
    Args:
        days_old: Number of days to consider sessions as old
    """
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Get old session IDs first
        cursor.execute("""
            SELECT id FROM chat_sessions 
            WHERE last_active < DATE_SUB(NOW(), INTERVAL %s DAY)
            AND status = TRUE
        """, (days_old,))
        
        old_session_ids = [row[0] for row in cursor.fetchall()]
        
        if old_session_ids:
            # Soft delete messages for old sessions
            placeholders = ','.join(['%s'] * len(old_session_ids))
            cursor.execute(f"""
                UPDATE messages 
                SET status = FALSE 
                WHERE session_id IN ({placeholders})
            """, old_session_ids)
            
            # Soft delete old sessions
            cursor.execute(f"""
                UPDATE chat_sessions 
                SET status = FALSE 
                WHERE id IN ({placeholders})
            """, old_session_ids)
            
            db.commit()
            return len(old_session_ids)
        
        return 0
        
    except Exception as e:
        db.rollback()
        raise e