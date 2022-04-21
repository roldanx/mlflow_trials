FROM python:3.9
WORKDIR /home
RUN ["pip", "install", "--upgrade", "pip"]
RUN ["pip", "install", "psycopg2-binary", "boto3", "mlflow==1.25.1"]
COPY load_env.sh load_env.sh
RUN ["chmod", "+x", "load_env.sh"]
EXPOSE 8000
ENTRYPOINT ["sh", "-c", "bash load_env.sh && mlflow server --backend-store-uri $DB_URI --default-artifact-root $MLRUNS_DIRECTORY -h 0.0.0.0 -p 5001"]
