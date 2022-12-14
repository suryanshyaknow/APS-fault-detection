import pendulum
import os
from airflow import DAG
from airflow.operators.python import PythonOperator


with DAG(
    'sensor_training',
    default_args={'retries': 2},
    description='APS-fault-detection',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2022, 14, 11, tz="UTC"),
    catchup=False,
    tags=['example']
) as dag:

    def training(**kwargs):
        from src.pipelines.training import TrainingPipeline
        TrainingPipeline().begin()

    def sync_artifact_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        os.system(f"aws s3 sync /app/artifact s3://{bucket_name}/artifacts")
        os.system(
            f"aws s3 sync /app/saved_models s3://{bucket_name}/saved_models")

    # Commence training
    training_pipeline = PythonOperator(
        task_id="training_pipeline", python_callable=training)

    # Sync the relevant objects to the S3 bucket
    sync_data_to_s3 = PythonOperator(
        task_id="sync_data_to_s3", python_callable=sync_artifact_to_s3_bucket)

    # flow
    training_pipeline >> sync_data_to_s3
