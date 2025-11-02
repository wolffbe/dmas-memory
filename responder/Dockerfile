FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY responder/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY dmas_llm /app/dmas_llm
COPY responder/responder.py /app/responder.py

EXPOSE 8006

CMD ["python", "responder.py"]