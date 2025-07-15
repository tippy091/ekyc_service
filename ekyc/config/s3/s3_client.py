import logging
import os

import boto3
from boto3.s3.transfer import TransferConfig

config = TransferConfig(use_threads=False)

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(self, static_config):
        self.s3_client = None
        self.bucket_export_excel = None
        self.public_endpoint = None

        self.create_client(static_config.asset_config['MINIO'])

    def create_client(self, asset_config):
        self.public_endpoint = asset_config.get("public_endpoint", os.environ.get('MINIO_PUBLIC_ENDPOINT'))
        self.bucket_export_excel = asset_config["export_excel_bucket"]

        endpoint = asset_config.get("endpoint", os.environ.get('MINIO_ENDPOINT'))
        access_key = asset_config.get("access_key", os.environ.get('MINIO_ACCESS_KEY'))
        secret_key = asset_config.get("secret_key", os.environ.get('MINIO_SECRET_KEY'))

        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

    def create_bucket_if_not_exists(self, bucket_name):
        s3 = self.s3_client
        try:
            existing_buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]
            if bucket_name not in existing_buckets:
                s3.create_bucket(Bucket=bucket_name)
                logger.info(f"Created bucket: {bucket_name}")
            else:
                logger.info(f"Bucket exists: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to check/create bucket {bucket_name}: {e}")

    def download_file(self, bucket_name, file_key, local_path):
        self.s3_client.download_file(bucket_name, file_key, local_path, Config=config)
        pass

    def upload_file(self, file_key, local_path, bucket_name="dev-ekyc", add_endpoint=True):
        object_name = os.path.basename(file_key)
        try:
            self.s3_client.upload_file(local_path, bucket_name, file_key, Config=config)
            logger.info(f"Uploading {object_name} to {bucket_name}")

            if add_endpoint:
                return f"{self.public_endpoint}/{bucket_name}/{file_key}"

            return f"{bucket_name}/{file_key}"
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
