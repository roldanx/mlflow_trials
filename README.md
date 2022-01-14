# mlflow_trials
> PREREQUISITES: 
>* To have **docker** installed.
>* To have **mlflow** and **psycopg2-binary** python packages installed.
>* To `cd` into the repo main folder (mlflow_trials).

Start up PostgreSQL for the Registry:
```
$ docker build --tag postgresql-mlflow postgresql 
$ docker run --name postgresql-mlflow -d --network host postgresql-mlflow 
$ mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow_db --default-artifact-root mlruns -h 0.0.0.0 -p 8000
```
Enter the database:
```
$ docker exec -it postgresql-mlflow sh
# psql --dbname mlflow_db -U mlflow
```
Train model
```
$ docker build --tag tree-classifier python
$ docker run --name tree-classifier -d tree-classifier 
```
