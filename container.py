import logging

from dependency_injector import containers, providers

from ekyc.config.mongodb.mongodb_client import MongoDBClient
from ekyc.config.s3.s3_client import S3Client
from ekyc.config.static_config import StaticConfig
from ekyc.repository.common_repository import CommonRepository
from ekyc.repository.ekyc_repository import EKYCRepository
from ekyc.repository.user_repository import UserRepository
from ekyc.service.common_service import CommonService
from ekyc.service.ekyc_service import EKYCService

logger = logging.getLogger(__name__)


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=[
        "ekyc.config",
        "ekyc.controller",
        "ekyc.service",
    ])

    config = providers.Configuration()

    static_config = providers.Singleton(
        StaticConfig,
        app_args=config.app_args
    )

    mongodb_client = providers.Singleton(
        MongoDBClient,
        static_config
    )

    s3_client = providers.Singleton(
        S3Client,
        static_config
    )

    common_repository = providers.Singleton(
        CommonRepository,
        mongodb_client
    )

    common_service = providers.Singleton(
        CommonService,
        s3_client,
        common_repository
    )

    user_repository = providers.Singleton(
        UserRepository,
        mongodb_client
    )

    ekyc_repository = providers.Singleton(
        EKYCRepository,
        mongodb_client
    )

    ekyc_service = providers.Singleton(
        EKYCService,
        s3_client,
        user_repository
    )
