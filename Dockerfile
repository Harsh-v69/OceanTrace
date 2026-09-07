# ---------------------------------------------------------------------------
# Hugging Face Spaces (Docker SDK) image for the OceanTrace prototype.
#
# CPU-only. Every ML model binary is committed to the repo, so there is no
# training step and no network access needed at run time. Build once (~10 min),
# then the Space serves the FastAPI app + Operations Console on port 7860.
# ---------------------------------------------------------------------------
FROM python:3.13-slim

# OpenCV (opencv-python-headless) still needs these shared libs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as uid 1000 ("user"). Match that so the app can
# write its SQLite file.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /home/user/app

# 1) CPU build of PyTorch from the dedicated index (keeps it ~200 MB, not ~2.5 GB
#    of CUDA). Pinned to the version this build was tested against.
RUN pip install --user torch==2.14.0 \
        --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else from PyPI. Heavy scientific stack pinned to the tested set.
COPY --chown=user requirements.txt .
RUN pip install --user \
        numpy==2.5.2 pandas==3.0.5 scikit-learn==1.9.0 \
        -r requirements.txt

# 3) The application (models, static console, code).
COPY --chown=user . .

# Writable dir for the SQLite database. On the free tier this is ephemeral -
# jurisdictions and the three role accounts are re-seeded on every boot, and the
# demo scenarios are deterministic, so a restart just clears ad-hoc investigations.
RUN mkdir -p data

EXPOSE 7860

# run.py -> uvicorn (single worker; the model cache is per-process).
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "7860"]
