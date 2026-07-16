FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# shared modules copied from the repo root at build context
COPY store.py ./store.py
COPY viewer ./viewer
COPY server ./server
RUN useradd -u 1001 -m appuser && mkdir -p /app/data /app/models \
    && chown -R 1001:1001 /app
USER 1001
ENV STORE_DB=/app/data/store.db STORE_DATA=/app/data \
    OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
EXPOSE 3000
# taskset pins FireRed (and everything) to core 1, leaving core 0 for acp/viewer.
CMD ["taskset", "-c", "1", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "3000"]
