import logging

from dependency_injector import containers, providers
from ekyc.service.common_service import CommonService
from ekyc.service.ekyc_service import EKYCService

logger = logging.getLogger(__name__)


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=[
        "ekyc.controller",
        "ekyc.service",
    ])
    config = providers.Configuration()
    common_service = providers.Singleton(
        CommonService,
    )

    ekyc_service = providers.Singleton(
        EKYCService,
    )
