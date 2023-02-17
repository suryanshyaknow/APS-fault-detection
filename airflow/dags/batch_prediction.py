import pendulum
import os
from airflow import DAG
from airflow.operators.python import PythonOperator

INPUT_DIR = "/app/input_files"


with DAG(
    'batch_prediction',
    default_args={'retries': 2},
    description='APS-fault-detection',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2023, 2, 17, tz="UTC"),
    catchup=False,
    tags=['batch_prediction']
) as dag:

    def download_input_files():
        bucket_name = os.getenv("BUCKET_NAME")
        # Make sure the input dir does exist
        os.makedirs(INPUT_DIR, exist_ok=True)
        # Syncing the input dir to S3 bucket
        os.system(
            f'aws s3 sync s3://{bucket_name}/input_files /app/input_files')

    def batch_prediction(**kwargs):
        from src.pipelines.batch_prediction import BatchPredictionPipeline
        for input_file in os.listdir(INPUT_DIR):
            BatchPredictionPipeline(input_file_path=input_file).initiate()

    def sync_predictions_dir_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        # Upload predictions folder to prediction_files dir in S3 bucket
        os.system(
            f"aws s3 sync /app/predictions s3://{bucket_name}/prediction_files")

    # Download input files
    download_input_files = PythonOperator(
        task_id="downloading_input_files", python_callable=download_input_files)

    # Sync the prediction files to the S3 bucket
    generate_prediction_files = PythonOperator(
        task_id="prediction_pipeline", python_callable=batch_prediction)

    # Upload prediction files
    upload_prediction_files = PythonOperator(
        task_id="uploading_prediciton_files", python_callable=sync_predictions_dir_to_s3_bucket)

    # flow
    download_input_files >> generate_prediction_files >> upload_prediction_files
