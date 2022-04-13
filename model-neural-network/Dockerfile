FROM python:3.9
WORKDIR /home
COPY model-neural-network/requirements.txt requirements.txt
RUN ["pip", "install", "--upgrade", "pip"]
RUN ["pip", "install", "-r", "requirements.txt"]
COPY load_env.sh load_env.sh
RUN ["chmod", "+x", "load_env.sh"]
COPY model-neural-network/neural-network.py neural-network.py 
RUN ["chmod", "+x", "neural-network.py"]
ENTRYPOINT ["sh", "-c", "bash load_env.sh && python neural-network.py"]
