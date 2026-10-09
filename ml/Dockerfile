FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /opt/ml
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && pip check

COPY settings.json .
COPY models ./models

EXPOSE 8080
CMD ["mlserver", "start", "/opt/ml"]
