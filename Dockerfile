FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory for API
RUN mkdir -p api/data_cache && chmod 777 api/data_cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose ports for API and Streamlit
EXPOSE 8000 8501

# Default command to run the API service
# You can override this when running the container with:
# podman run -p 8501:8501 your-image-name streamlit run frontend/main.py --server.port 8501 --server.address 0.0.0.0
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
