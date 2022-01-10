FROM python:3.9
WORKDIR /home/roldanx/code/tfm/python
COPY requirements.txt requirements.txt
COPY decision-tree-classifier.py decision-tree-classifier.py 
COPY car_evaluation.csv car_evaluation.csv 
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
