import os
import boto3
from botocore.exceptions import ClientError

def main():
    # Get S3 credentials and endpoint from environment variables
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    bucket_name = os.environ.get("MLFLOW_S3_BUCKET", "mlflow-artifacts")
    test_key = "test-mlflow-artifact.txt"
    test_content = b"MLflow S3 connection test."

    if not aws_access_key_id or not aws_secret_access_key or not s3_endpoint_url:
        print("Missing one or more required environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, MLFLOW_S3_ENDPOINT_URL")
        return

    # Create S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Try to upload a test file
    try:
        print(f"Uploading test file to bucket '{bucket_name}' at {s3_endpoint_url} ...")
        s3.put_object(Bucket=bucket_name, Key=test_key, Body=test_content)
        print("✅ S3 upload succeeded.")
    except ClientError as e:
        print("❌ S3 upload failed:", e)
        return

    # Try to download the test file
    try:
        print(f"Downloading test file from bucket '{bucket_name}' ...")
        response = s3.get_object(Bucket=bucket_name, Key=test_key)
        content = response["Body"].read()
        if content == test_content:
            print("✅ S3 download succeeded and content matches.")
        else:
            print("❌ S3 download succeeded but content does not match.")
    except ClientError as e:
        print("❌ S3 download failed:", e)
        return

    # Clean up: delete the test file
    try:
        print(f"Deleting test file from bucket '{bucket_name}' ...")
        s3.delete_object(Bucket=bucket_name, Key=test_key)
        print("✅ S3 cleanup succeeded.")
    except ClientError as e:
        print("❌ S3 cleanup failed:", e)

if __name__ == "__main__":
    main()