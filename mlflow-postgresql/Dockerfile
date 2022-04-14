FROM postgres:14.1
ENV POSTGRES_DB mlflow_db
ENV POSTGRES_USER mlflow
ENV POSTGRES_PASSWORD mlflow
EXPOSE 5432
COPY mlflow-postgresql/init-user-db.sql /docker-entrypoint-initdb.d/
