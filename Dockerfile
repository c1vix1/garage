FROM python:3.10-slim

# Enable Debian non-free-firmware repository (required for Intel GPU OpenCL drivers)
RUN echo "deb http://deb.debian.org/debian bookworm main non-free-firmware" > /etc/apt/sources.list.d/non-free.list

# Install OpenCV and Intel GPU dependencies
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