import json
import os

from pymongo import MongoClient

mongodb_clients = {}


def create_clients(asset_config):
    mongodb_config = asset_config["MONGODB"]
    clients = json.loads(mongodb_config["clients"])

    for client in clients:
        client_config = asset_config[client]

        host = client_config.get('host', os.environ.get('MONGO_ENDPOINT'))
        port = int(client_config.get('port', os.environ.get('MONGO_PORT')))
        database_name = client_config.get('database', os.environ.get('MONGO_DATABASE'))
        username = client_config.get('username', os.environ.get('MONGO_USER'))
        password = client_config.get('password', os.environ.get('MONGO_PASS'))

        mongodb_client = MongoClient(host, port, username=username, password=password)

        mongodb_clients[client] = mongodb_client[database_name]
    pass


class MongoDBClient:
    def __init__(self, static_config):
        self.asset_config = static_config.asset_config
        create_clients(self.asset_config)
        mongodb_config = self.asset_config["MONGODB"]
        self.default_client = mongodb_config.get("clients")

        pass

    

    def get_mongodb_client(self, client_name):
        return mongodb_clients[client_name]
    

    def get_collection(self, client_name: str, collection_name: str):
        db = self.get_mongodb_client(client_name)
        return db[collection_name]