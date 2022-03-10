FROM python:3.9
WORKDIR /home
COPY requirements.txt requirements.txt
COPY neural-network.py neural-network.py 
RUN ["chmod", "+x", "neural-network.py"]
RUN ["mkdir", "/home/mlruns"]
RUN ["pip", "install", "--upgrade", "pip"]
RUN ["pip", "install", "-r", "requirements.txt"]
ENTRYPOINT ["python", "neural-network.py"]
