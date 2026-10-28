FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit==1.38.0 requests

COPY ui.py .

CMD ["streamlit", "run", "ui.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
