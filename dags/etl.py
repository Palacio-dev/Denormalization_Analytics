from airflow.sdk import dag, task
from pendulum import datetime

@dag(
    dag_id='etl_dag',
    start_date=datetime(2024, 1, 1),
    schedule=None, # Para você disparar manualmente no botão 'Play'
    catchup=False
)
def etl_dag():

    @task
    def extract():
        print('Extracting data...')
        return {"data": [1,2,3,4,5]}

    @task
    def transform(extracted_data):
        print('Transforming data...')
        transformed_data = [x*10 for x in extracted_data['data']]
        return {"transformed_data": transformed_data}

    @task
    def load(transformed_data):
        print('Loading data...')
        print(transformed_data)

    t1 = extract()
    t2 = transform(t1)
    t3 = load(t2)

    t1 >> t2 >> t3



etl_dag()