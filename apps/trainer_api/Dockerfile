FROM python:3.11-slim
WORKDIR /app/apps/trainer_api
# git is required: requirements.txt pins mobile-sam / sam-2 as git+https deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY apps/trainer_api/requirements.txt /app/apps/trainer_api/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV SEG_HOST=0.0.0.0
CMD uvicorn app.main:app --host ${SEG_HOST} --port 8000
