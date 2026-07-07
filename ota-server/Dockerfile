FROM python:3.11-slim

WORKDIR /app

# Dépendances
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# Code source
COPY main.py .

# Répertoire firmware (monté via volume dans docker-compose)
RUN mkdir -p /firmware/history

# Health check natif Docker
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8090/ || exit 1

EXPOSE 8090

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090", "--log-level", "info"]
