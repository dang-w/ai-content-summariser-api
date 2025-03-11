FROM python:3.9-slim

WORKDIR /app

# Create a writable cache directory
RUN mkdir -p /tmp/huggingface_cache && \
  chmod 777 /tmp/huggingface_cache

# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache
ENV HF_HOME=/tmp/huggingface_cache
ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
