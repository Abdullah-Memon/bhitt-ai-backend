from flask import Blueprint, jsonify, request, Response, g
from ..utils.auth import token_required
from ..utils.limiter import limiter
from ..models.poetry.search_engine import query_sindhi_poetry, query_sindhi_poetry_with_session
from ..models.general_chatbot.chatbot import query_general_chatbot, query_general_chatbot_with_session
from ..services.session_service import SessionService
import logging

logger = logging.getLogger(__name__)

def safe_log(logger_func, message):
    """Safely log messages that may contain Unicode characters"""
    try:
        logger_func(message)
    except UnicodeEncodeError:
        # Replace non-ASCII characters with their Unicode escape sequences
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        logger_func(f"[Unicode content] {safe_message}")

new_models_bp = Blueprint('new_models', __name__)

@new_models_bp.route('/poetry/<path:query>', methods=['GET'])
@limiter.limit("100 per minute")
@token_required
def poetry_search(query):
    """
    Poetry search endpoint
    GET /poetry/{query}
    """
    try:
        if not query or not isinstance(query, str) or not query.strip():
            return jsonify({'error': 'Query must be a non-empty string'}), 400
        
        # Safely log query with Unicode characters
        safe_log(logger.info, f"Poetry search query: {query}")
        
        # Call the poetry search function
        response = query_sindhi_poetry(query.strip())
        
        return jsonify({
            'query': query,
            'response': response,
            'model': 'poetry-search-engine'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in poetry search: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@new_models_bp.route('/chatbot/<path:query>', methods=['GET'])
@limiter.limit("100 per minute")
@token_required
def general_chatbot(query):
    """
    General chatbot endpoint
    GET /chatbot/{query}
    """
    try:
        if not query or not isinstance(query, str) or not query.strip():
            return jsonify({'error': 'Query must be a non-empty string'}), 400
        
        # Safely log query with Unicode characters
        safe_log(logger.info, f"General chatbot query: {query}")
        
        # Call the general chatbot function
        response = query_general_chatbot(query.strip())
        
        return jsonify({
            'query': query,
            'response': response,
            'model': 'general-chatbot'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in general chatbot: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@new_models_bp.route('/poetry', methods=['POST'])
@limiter.limit("100 per minute")
@token_required
def poetry_search_post():
    """
    Poetry search endpoint with POST method and session management
    POST /poetry
    Body: {"query": "search query", "session_id": "optional session id"}
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        if not query or not isinstance(query, str) or not query.strip():
            return jsonify({'error': 'Query must be a non-empty string'}), 400
        
        session_id = data.get('session_id', None)
        user_id = g.user_id  # From token_required decorator
        
        # Safely log query with Unicode characters
        safe_log(logger.info, f"Poetry search query (POST): {query}, user: {user_id}, session: {session_id}")
        
        # Call session-aware poetry search function
        result = query_sindhi_poetry_with_session(query.strip(), user_id, session_id)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in poetry search (POST): {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@new_models_bp.route('/chatbot', methods=['POST'])
@limiter.limit("100 per minute")
@token_required
def general_chatbot_post():
    """
    General chatbot endpoint with POST method and session management
    POST /chatbot
    Body: {"query": "user question", "session_id": "optional session id"}
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        if not query or not isinstance(query, str) or not query.strip():
            return jsonify({'error': 'Query must be a non-empty string'}), 400
        
        session_id = data.get('session_id', None)
        user_id = g.user_id  # From token_required decorator
        
        # Safely log query with Unicode characters
        safe_log(logger.info, f"General chatbot query (POST): {query}, user: {user_id}, session: {session_id}")
        
        # Call session-aware general chatbot function
        result = query_general_chatbot_with_session(query.strip(), user_id, session_id)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in general chatbot (POST): {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@new_models_bp.route('/models/health', methods=['GET'])
def models_health():
    """
    Health check endpoint for the new models
    """
    try:
        return jsonify({
            'status': 'healthy',
            'models': {
                'poetry-search-engine': 'available',
                'general-chatbot': 'available'
            },
            'endpoints': [
                'GET /poetry/{query}',
                'POST /poetry',
                'GET /chatbot/{query}',
                'POST /chatbot',
                'GET /sessions',
                'GET /sessions/{session_id}/messages',
                'DELETE /sessions/{session_id}'
            ]
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Session Management Endpoints

@new_models_bp.route('/sessions', methods=['GET'])
@limiter.limit("50 per minute")
@token_required
def get_user_sessions():
    """
    Get all sessions for the authenticated user
    GET /sessions
    """
    try:
        user_id = g.user_id
        limit = request.args.get('limit', 20, type=int)
        
        if limit > 100:  # Prevent excessive data retrieval
            limit = 100
        
        sessions = SessionService.get_user_sessions(user_id, limit)
        
        return jsonify({
            'sessions': sessions,
            'count': len(sessions)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting user sessions: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@new_models_bp.route('/sessions/<session_id>/messages', methods=['GET'])
@limiter.limit("50 per minute")
@token_required
def get_session_messages(session_id):
    """
    Get messages for a specific session
    GET /sessions/{session_id}/messages
    """
    try:
        user_id = g.user_id
        limit = request.args.get('limit', 50, type=int)
        
        if limit > 200:  # Prevent excessive data retrieval
            limit = 200
        
        # Verify session belongs to user
        if not SessionService.verify_session_belongs_to_user(session_id, user_id):
            return jsonify({
                'error': 'Session not found or unauthorized',
                'code': 'SESSION_UNAUTHORIZED'
            }), 404
        
        messages = SessionService.get_session_messages(session_id, limit)
        session = SessionService.get_session(session_id)
        
        return jsonify({
            'session': session,
            'messages': messages,
            'count': len(messages)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting session messages: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@new_models_bp.route('/sessions/<session_id>', methods=['DELETE'])
@limiter.limit("20 per minute")
@token_required
def delete_session(session_id):
    """
    Delete a specific session
    DELETE /sessions/{session_id}
    """
    try:
        user_id = g.user_id
        
        success = SessionService.delete_session(session_id, user_id)
        
        if not success:
            return jsonify({
                'error': 'Session not found or unauthorized',
                'code': 'SESSION_UNAUTHORIZED'
            }), 404
        
        return jsonify({
            'message': 'Session deleted successfully',
            'session_id': session_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500
