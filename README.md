# Running, saving and accessing tree classifier model on MLFlow Registry (LOCAL DEPLOYEMNT)
> PREREQUISITES: 
>* To have **docker** installed.
>* To have **mlflow** and **psycopg2-binary** python packages installed.

Define artifacts storage location:
```
$ export MLRUNS=<PATH_TO_MLRUNS>
```

Start up PostgreSQL:
```
$ docker run --name postgresql-mlflow -d --network host roldanx/postgresql-mlflow
```
Start up Tracking server:
```
$ mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow_db --default-artifact-root $MLRUNS -h 0.0.0.0 -p 8000
```
Train model:
```
$ docker run --name <PROJECT> --network host -v $MLRUNS:$MLRUNS roldanx/<PROJECT>
# EXAMPLE:
# podman run --name decision-tree --network host -v $MLRUNS:$MLRUNS:Z roldanx/tree-classifier-mlflow
```
*info: volume hostpath and mountpoint must be the same, since the container will store the artifacts wherever the mlflow "artifact root" is set.
<br>

Check logged tree classifier model (from `browser`):

```
http://localhost:8000
```
<br>

# Optional steps
Manual docker build (from `mlflow_trials` folder):
```
$ docker build --tag postgresql-mlflow postgresql
$ docker build --tag <PROJECT> model-<PROJECT>
```
Enter the database:
```
$ docker exec -it postgresql-mlflow sh
# psql --dbname mlflow_db -U mlflow
```
