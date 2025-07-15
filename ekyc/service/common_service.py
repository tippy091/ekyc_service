from ekyc.config.s3.s3_client import S3Client
from ekyc.repository.common_repository import CommonRepository


class CommonService:
    def __init__(self, s3_client: S3Client, common_repository: CommonRepository):
        self.s3_client = s3_client
        self.common_repository = common_repository

        pass

    def get_status(self):
        return {"status": "ok", "version": "1.0.0"}
