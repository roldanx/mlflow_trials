FROM python:3.9
WORKDIR /home
COPY model-tree-classifier/requirements.txt requirements.txt
RUN ["pip", "install", "--upgrade", "pip"]
RUN ["pip", "install", "-r", "requirements.txt"]
COPY load_env.sh load_env.sh
RUN ["chmod", "+x", "load_env.sh"]
COPY model-tree-classifier/decision-tree-classifier.py decision-tree-classifier.py 
COPY model-tree-classifier/car_evaluation.csv car_evaluation.csv 
RUN ["chmod", "+x", "decision-tree-classifier.py"]
ENTRYPOINT ["sh", "-c", "bash load_env.sh && python decision-tree-classifier.py"]
