from flask import Blueprint, request, jsonify
from ..utils.auth import token_required
from ..services.subscription_service import get_subscription_details, set_subscription
from ..utils.limiter import limiter

subscription_bp = Blueprint('subscription', __name__)

@subscription_bp.route('/subscription-details', methods=['GET'])
@limiter.limit("100 per minute")
@token_required
def subscription_details():
    response, status = get_subscription_details()
    return jsonify(response), status

@subscription_bp.route('/set-subscription', methods=['POST'])
@limiter.limit("100 per minute")
@token_required
def set_subscription_endpoint():
    data = request.get_json()
    response, status = set_subscription(data)
    return jsonify(response), status