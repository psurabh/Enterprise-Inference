# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    sox \
    libsox-fmt-all \
    libsox-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    numpy==1.26.4 \
    && pip install --no-cache-dir fairseq==0.12.2 \
    && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY samples/ ./samples/
COPY hindi.pt .

# Set environment variables
ENV DEFAULT_SAMPLING_RATE=16000
ENV PYTHONUNBUFFERED=1

# Default command to run the ASR service
CMD ["python", "pod_service.py"]
