# Running, saving and accessing tree classifier model on MLFlow Registry
> PREREQUISITES: 
>* To have **docker** installed.
>* To have **mlflow** and **psycopg2-binary** python packages installed.

Start up PostgreSQL:
```
$ docker run --name postgresql-mlflow -d --network host roldanx/postgresql-mlflow
```
Start up Tracking server:
```
$ mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow_db --default-artifact-root mlruns -h 0.0.0.0 -p 8000
```
Train model:
```
$ docker run --name tree-classifier-mlflow --network host roldanx/tree-classifier-mlflow
```
Check logged tree classifier model (from `browser`):

```
http://localhost:8000
```
<br>

# Optional steps
Manual docker build (from `mlflow_trials` folder):
```
$ docker build --tag postgresql-mlflow postgresql
$ docker build --tag tree-classifier-mlflow python
```
Enter the database:
```
$ docker exec -it postgresql-mlflow sh
# psql --dbname mlflow_db -U mlflow
```
