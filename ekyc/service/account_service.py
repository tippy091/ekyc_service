from ekyc.config.s3.s3_client import S3Client
from ekyc.repository.user_repository import UserRepository

class AccountService:
    def __init__(self, s3_client: S3Client, user_repository: UserRepository):
        self.s3_client = s3_client
        self.user_repository = user_repository
        pass


