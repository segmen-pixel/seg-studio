FROM python:3.11-slim
WORKDIR /app/apps/serving_api
COPY apps/serving_api/requirements.txt /app/apps/serving_api/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV SEG_HOST=0.0.0.0
CMD uvicorn app.main:app --host ${SEG_HOST} --port 8001
