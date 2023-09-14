# syntax = docker/dockerfile:experimental
FROM python:3.9.7

WORKDIR /app

# Python dependencies
COPY requirements.txt ./
RUN pip3 --no-cache-dir install -r requirements.txt

COPY . ./

ENTRYPOINT [ "python3",  "generate.py" ]