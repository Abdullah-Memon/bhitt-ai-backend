from flask import Blueprint, jsonify
from ..utils.auth import token_required
from ..services.user_service import get_user_details
from ..utils.limiter import limiter

user_bp = Blueprint('user', __name__)

@user_bp.route('/user-details', methods=['GET'])
@limiter.limit("100 per minute")
@token_required
def user_details():
    response, status = get_user_details()
    return jsonify(response), status