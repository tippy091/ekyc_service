from ekyc.config.mongodb.mongodb_client import MongoDBClient


class EKYCRepository:
    def __init__(self, mongodb_client: MongoDBClient):
        self.mongodb_client = mongodb_client
        pass