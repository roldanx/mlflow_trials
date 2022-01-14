# mlflow_trials
Start up PostgreSQL for the Registry:
```
$ docker build --tag postgresql-mlflow postgresql 
$ docker run --name postgresql-mlflow -d --network host postgresql-mlflow 
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
