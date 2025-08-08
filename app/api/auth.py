from flask import Blueprint, request, jsonify
from ..services.auth_service import login_user, register_user, verify_token
from ..utils.limiter import limiter

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['POST'])
@limiter.limit("100 per minute")
def login():
    data = request.get_json()
    response, status = login_user(data)
    return jsonify(response), status

@auth_bp.route('/register', methods=['POST'])
@limiter.limit("100 per minute")
def register():
    data = request.get_json()
    response, status = register_user(data)
    return jsonify(response), status