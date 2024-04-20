FROM python:3.11-alpine
LABEL maintainer="HS Cuban Team"

RUN apk add --no-cache python3-dev openssl-dev libffi-dev libc-dev gcc \
                        libsodium-dev curl && pip3 install --upgrade pip

COPY . /app
WORKDIR /app

ENV PIP_ROOT_USER_ACTION=ignore \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN /root/.local/bin/poetry install --no-interaction --no-ansi

CMD python -m nepherite