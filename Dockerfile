FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY web ./web
COPY scripts ./scripts
COPY schema.sql ./

# Config comes from real environment variables in production
# (DATABASE_URL, GOOGLE_API_KEY, ...). No .env is copied into the image.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
