FROM python:3.9
WORKDIR /home/roldanx/code/tfm/python
COPY requirements.txt requirements.txt
COPY decision-tree-classifier.py decision-tree-classifier.py 
COPY car_evaluation.csv car_evaluation.csv 
RUN ["chmod", "+x", "decision-tree-classifier.py"]
RUN ["pip", "install", "--upgrade", "pip"]
RUN ["pip", "install", "-r", "requirements.txt"]
ENTRYPOINT ["python", "decision-tree-classifier.py"]
