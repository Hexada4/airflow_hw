import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

path = os.path.expanduser('~/airflow_hw')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)

from modules.pipeline import pipeline
from modules.predict import predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 8, 11),
    'retries': 1,
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:
    first_task = BashOperator(
        task_id = 'introduction',
        bash_command="echo '[INFO] Dag started!'"
    )

    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
    )

    predict = PythonOperator(
        task_id='predictor',
        python_callable=predict
    )

    end = BashOperator(
        task_id='end',
        bash_command="echo '[INFO] Dag completed tasks!'"
    )

    first_task >> pipeline >> predict >> end