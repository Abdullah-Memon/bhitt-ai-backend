import uuid
from datetime import datetime
from flask import g
from ..utils.database import get_db
import logging

logger = logging.getLogger(__name__)

class SessionService:
    @staticmethod
    def create_session(user_id: str, session_name: str) -> str:
        """
        Create a new chat session for a user
        
        Args:
            user_id: The ID of the user
            session_name: Name for the session (usually the first query)
            
        Returns:
            session_id: The ID of the newly created session
        """
        try:
            db = get_db()
            cursor = db.cursor()
            
            session_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            # Truncate session name if it's too long
            if len(session_name) > 255:
                session_name = session_name[:252] + "..."
            
            cursor.execute("""
                INSERT INTO chat_sessions (id, user_id, session_name, status, created_at, last_active)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (session_id, user_id, session_name, True, current_time, current_time))
            
            db.commit()
            logger.info(f"Created new session {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            db.rollback()
            raise e
    
    @staticmethod
    def get_session(session_id: str) -> dict:
        """
        Get session details by session ID
        
        Args:
            session_id: The ID of the session
            
        Returns:
            dict: Session details or None if not found
        """
        try:
            db = get_db()
            cursor = db.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT id, user_id, session_name, status, created_at, last_active
                FROM chat_sessions
                WHERE id = %s AND status = TRUE
            """, (session_id,))
            
            session = cursor.fetchone()
            return session
            
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {str(e)}")
            return None
    
    @staticmethod
    def update_session_activity(session_id: str):
        """
        Update the last_active timestamp for a session
        
        Args:
            session_id: The ID of the session
        """
        try:
            db = get_db()
            cursor = db.cursor()
            
            cursor.execute("""
                UPDATE chat_sessions
                SET last_active = %s
                WHERE id = %s AND status = TRUE
            """, (datetime.now(), session_id))
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error updating session activity for {session_id}: {str(e)}")
    
    @staticmethod
    def save_message(session_id: str, sender: str, message: str, stream_chunk: bool = False) -> str:
        """
        Save a message to the database
        
        Args:
            session_id: The ID of the session
            sender: 'user' or 'bot'
            message: The message content
            stream_chunk: Whether this is a streaming chunk
            
        Returns:
            message_id: The ID of the saved message
        """
        try:
            db = get_db()
            cursor = db.cursor()
            
            message_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            cursor.execute("""
                INSERT INTO messages (id, session_id, sender, message, stream_chunk, status, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (message_id, session_id, sender, message, stream_chunk, True, current_time))
            
            db.commit()
            logger.info(f"Saved {sender} message {message_id} to session {session_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            db.rollback()
            raise e
    
    @staticmethod
    def get_session_messages(session_id: str, limit: int = 50) -> list:
        """
        Get messages for a session
        
        Args:
            session_id: The ID of the session
            limit: Maximum number of messages to retrieve
            
        Returns:
            list: List of messages
        """
        try:
            db = get_db()
            cursor = db.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT id, session_id, sender, message, stream_chunk, status, timestamp
                FROM messages
                WHERE session_id = %s AND status = TRUE
                ORDER BY timestamp ASC
                LIMIT %s
            """, (session_id, limit))
            
            messages = cursor.fetchall()
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages for session {session_id}: {str(e)}")
            return []
    
    @staticmethod
    def get_user_sessions(user_id: str, limit: int = 20) -> list:
        """
        Get all sessions for a user
        
        Args:
            user_id: The ID of the user
            limit: Maximum number of sessions to retrieve
            
        Returns:
            list: List of sessions
        """
        try:
            db = get_db()
            cursor = db.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT id, user_id, session_name, status, created_at, last_active
                FROM chat_sessions
                WHERE user_id = %s AND status = TRUE
                ORDER BY last_active DESC
                LIMIT %s
            """, (user_id, limit))
            
            sessions = cursor.fetchall()
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting sessions for user {user_id}: {str(e)}")
            return []
    
    @staticmethod
    def verify_session_belongs_to_user(session_id: str, user_id: str) -> bool:
        """
        Verify that a session belongs to a specific user
        
        Args:
            session_id: The ID of the session
            user_id: The ID of the user
            
        Returns:
            bool: True if session belongs to user, False otherwise
        """
        try:
            db = get_db()
            cursor = db.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM chat_sessions
                WHERE id = %s AND user_id = %s AND status = TRUE
            """, (session_id, user_id))
            
            result = cursor.fetchone()
            return result[0] > 0
            
        except Exception as e:
            logger.error(f"Error verifying session ownership: {str(e)}")
            return False
    
    @staticmethod
    def delete_session(session_id: str, user_id: str) -> bool:
        """
        Soft delete a session (mark as inactive)
        
        Args:
            session_id: The ID of the session
            user_id: The ID of the user (for security)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Verify session belongs to user
            if not SessionService.verify_session_belongs_to_user(session_id, user_id):
                return False
            
            db = get_db()
            cursor = db.cursor()
            
            # Soft delete session
            cursor.execute("""
                UPDATE chat_sessions
                SET status = FALSE
                WHERE id = %s AND user_id = %s
            """, (session_id, user_id))
            
            # Soft delete associated messages
            cursor.execute("""
                UPDATE messages
                SET status = FALSE
                WHERE session_id = %s
            """, (session_id,))
            
            db.commit()
            logger.info(f"Deleted session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            db.rollback()
            return False
