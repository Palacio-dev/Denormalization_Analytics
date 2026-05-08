from airflow.sdk import dag, task
from datetime import datetime

@dag(
    dag_id='xcoms_dag_kwargs',
    start_date=datetime(2024, 1, 1),
    schedule=None, # Para você disparar manualmente no botão 'Play'
    catchup=False
)
def xcoms_dag_kwargs():

    @task
    def first_task(**kwargs):
        #Extracting t1 for kwargs to push xComs manually
        ti = kwargs['ti']
        print('Extracting data ... This is the first task')
        fetched_data = {"data": [1,2,3,4,5]}
        ti.xcom_push(key='return_result', value=fetched_data)
    
    @task
    def second_task(**kwargs):
        #Extracting t2 for kwargs to pull xComs manually
        ti = kwargs['ti']
        fetched_data = ti.xcom_pull(task_ids='first_task', key='return_result')['data']
        print('Transforming data ... This is the second task')
        transformed_data = fetched_data * 2
        transformed_data = {"transformed_data": transformed_data}
        ti.xcom_push(key='return_result', value=transformed_data)


    @task
    def third_task(**kwargs):
        #Extracting t3 for kwargs to pull xComs manually
        ti = kwargs['ti']
        load_data = ti.xcom_pull(task_ids='second_task', key='return_result')
        print('Loading data ... This is the third task')
        return load_data

    # No Airflow 3/TaskFlow, você pode chamar as tasks assim:
    t1 = first_task()
    t2 = second_task()
    t3 = third_task()

    t1 >> t2 >> t3 

xcoms_dag_kwargs()