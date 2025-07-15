import logging

from dependency_injector.wiring import Provide, inject
from flask import jsonify

from blueprint import api_bp
from container import Container
from ekyc.service.common_service import CommonService

logger = logging.getLogger(__name__)


@api_bp.route("/health-check")
@inject
def health_check(common_service: CommonService = Provide[Container.common_service]):
    logger.info("I'm still alive")
    return jsonify(common_service.get_status()), 200


