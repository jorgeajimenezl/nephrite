FROM python:3.11-alpine
LABEL maintainer="HS Cuban Team <jorgeajimenezl17@gmail.com>"

RUN apk add --no-cache python3-dev openssl-dev libffi-dev libc-dev gcc libsodium-dev && pip3 install --upgrade pip

COPY . /app
WORKDIR /app

ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install poetry
RUN poetry install

CMD python -m nepherite