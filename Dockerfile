FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and frontend
COPY *.py ./
COPY index.html ./

# Copy runtime dataset
COPY final_processed_data.json ./

# Cache HuggingFace models in a dedicated directory
ENV HF_HOME=/app/.hf_cache
ENV TRANSFORMERS_CACHE=/app/.hf_cache
ENV PORT=25565

EXPOSE ${PORT}

CMD ["python", "web_app.py"]
