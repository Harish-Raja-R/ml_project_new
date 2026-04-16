# NeuroCollab v3.0 MAX — Dockerfile
# Supports Flask, Streamlit, and Gradio in one image.
#
# Build:
#   docker build -t neurocollab .
#
# Run Flask API (port 5000):
#   docker run -p 5000:5000 -e APP=flask neurocollab
#
# Run Streamlit (port 8501):
#   docker run -p 8501:8501 -e APP=streamlit neurocollab
#
# Run Gradio (port 7860):
#   docker run -p 7860:7860 -e APP=gradio neurocollab

FROM python:3.11-slim

LABEL maintainer="NeuroCollab v3.0 MAX"
LABEL description="Academic Collaboration Link Prediction — Graph ML"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn gradio streamlit plotly

# Copy project
COPY . .

# Create empty models dir if absent
RUN mkdir -p models plots

# Expose all possible ports
EXPOSE 5000 7860 8501 10000

# Entrypoint script
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV APP=flask
ENV PORT=5000

ENTRYPOINT ["/entrypoint.sh"]
