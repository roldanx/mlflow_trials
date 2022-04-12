FROM python:3.9
WORKDIR /home
COPY app/requirements.txt requirements.txt
RUN ["pip", "install", "--upgrade", "pip"]
RUN ["pip", "install", "-r", "requirements.txt"]
COPY load_env.sh load_env.sh
RUN ["chmod", "+x", "load_env.sh"]
COPY app/templates templates
COPY app/app.py app.py
RUN ["chmod", "+x", "app.py"]
ENTRYPOINT ["sh", "-c", "bash load_env.sh && python app.py"]
