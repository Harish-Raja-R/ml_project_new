#!/bin/bash
# docker-entrypoint.sh — NeuroCollab

set -e

case "$APP" in
  flask)
    echo "🚀 Starting Flask API on port ${PORT:-5000}..."
    exec gunicorn flask_app.app:app \
      --workers 2 \
      --timeout 120 \
      --bind "0.0.0.0:${PORT:-5000}"
    ;;
  streamlit)
    echo "🎨 Starting Streamlit UI on port ${PORT:-8501}..."
    exec streamlit run streamlit_app.py \
      --server.port "${PORT:-8501}" \
      --server.address 0.0.0.0 \
      --server.headless true \
      --browser.gatherUsageStats false
    ;;
  gradio)
    echo "🤗 Starting Gradio UI on port ${PORT:-7860}..."
    GRADIO_SERVER_PORT="${PORT:-7860}" GRADIO_SERVER_NAME="0.0.0.0" \
    exec python gradio_app.py
    ;;
  *)
    echo "Usage: APP=[flask|streamlit|gradio] docker run ..."
    exit 1
    ;;
esac
