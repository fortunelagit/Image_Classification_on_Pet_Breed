FROM python:3.11-slim

ENV PYTHONUNBUFFERED True

RUN pip install --upgrade pip

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 0 main:app
