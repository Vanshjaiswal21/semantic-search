FROM python:3.11-slim

# Hugging Face runs containers as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/hf \
    API_URL=http://127.0.0.1:8000
WORKDIR $HOME/app

# CPU-only PyTorch is much smaller than the default build
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-deploy.txt

# Download both models at build time so the app starts faster
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user data ./data

EXPOSE 7860

# API in the background, UI in the foreground on port 7860
CMD uvicorn app.api:app --host 0.0.0.0 --port 8000 & \
    streamlit run app/ui.py --server.port 7860 --server.address 0.0.0.0 \
    --server.headless true --server.enableCORS false --server.enableXsrfProtection false