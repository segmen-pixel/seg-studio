FROM python:3.11-slim
WORKDIR /app/apps/trainer_api
# git is required: requirements.txt pins mobile-sam / sam-2 as git+https deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libvips42 libpango-1.0-0 libpangoft2-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY apps/trainer_api/requirements.txt /app/apps/trainer_api/requirements.txt
COPY apps/serving_api/requirements.txt /app/apps/serving_api/requirements.txt
# The trainer lockfile leaves onnxruntime out on purpose; the serving lockfile pins it.
RUN pip install --no-cache-dir -r requirements.txt -r /app/apps/serving_api/requirements.txt
# Files Seg-Studio may not ship (a non-commercial torchmetrics module; pyphen
# dictionaries that are GPL-only, unclearly licensed or unlicensed): the table
# every install route applies.
COPY scripts/_nc_stubs.py /app/scripts/_nc_stubs.py
RUN python /app/scripts/_nc_stubs.py
COPY . /app
ENV SEG_HOST=0.0.0.0
CMD uvicorn app.main:app --host ${SEG_HOST} --port 8000
