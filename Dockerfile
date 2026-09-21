FROM python:3.10-slim

# Install system dependencies required for OpenCV and OpenVINO iGPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
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