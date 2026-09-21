FROM python:3.10-slim

# Install system dependencies for OpenCV and Intel GPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    clinfo \
    intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY models/ ./models/

CMD ["python", "-u", "main.py"]