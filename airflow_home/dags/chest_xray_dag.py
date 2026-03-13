from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add src to path so Airflow can find our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import (
    validate_data,
    preprocess_data,
    augment_data,
    split_data,
    version_data
)

# ── DAG default arguments ────────────────────────────────────────────────
default_args = {
    "owner": "chest_xray_project",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# ── Define the DAG ───────────────────────────────────────────────────────
dag = DAG(
    "chest_xray_ingestion_pipeline",
    default_args=default_args,
    description="Automated pipeline: validate → preprocess → augment → split → version",
    schedule_interval="@daily",
    catchup=False,
    tags=["chest-xray", "medical-imaging", "data-engineering"],
)

# ── Define tasks ─────────────────────────────────────────────────────────
task_validate = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag,
)

task_augment = PythonOperator(
    task_id="augment_data",
    python_callable=augment_data,
    dag=dag,
)

task_split = PythonOperator(
    task_id="split_data",
    python_callable=split_data,
    dag=dag,
)

task_version = PythonOperator(
    task_id="version_data",
    python_callable=version_data,
    dag=dag,
)

# ── Set task dependencies (the pipeline order) ───────────────────────────
task_validate >> task_preprocess >> task_augment >> task_split >> task_version