from ekyc.config.mongodb.mongodb_client import MongoDBClient


class UserRepository:
    def __init__(self, mongodb_client: MongoDBClient):
        self.collection = mongodb_client.get_collection("mongodb_amc_backend", "user_ekyc_verifications")

    def save_verification(self, data: dict):
        return self.collection.insert_one(data)
    