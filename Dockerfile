FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir  -r /code/requirements.txt

COPY . /code

RUN pip install --no-cache-dir gdown \
    && cd /code/finetuned_models \
    && gdown https://drive.google.com/uc?id=1qNjTCwMG9Km7Iiy3tLvayLPC6WKloMMj \
    && cd /code

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]