# ==============================================================================
# Kallvettu — Ancient Inscription Restoration & Translation Platform
# Production Multi-Stage / Hardened Dockerfile
# Base: Python 3.12 Slim with Tamil Font Rendering & Non-Root Execution
# ==============================================================================

FROM python:3.12-slim AS runtime

# Set non-interactive debian frontend and python environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install system dependencies, curl for healthchecks, and Tamil Noto fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    fonts-noto-core \
    fonts-noto-extra \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group (UID/GID 1000)
RUN groupadd -g 1000 kallvettu && \
    useradd -u 1000 -g kallvettu -m -s /bin/bash kallvettu

# Set working directory
WORKDIR /app

# Install Python dependencies first (leverage Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY --chown=kallvettu:kallvettu . .

# Switch to non-root execution
USER kallvettu

# Expose standard Streamlit port
EXPOSE 8501

# Container healthcheck using Streamlit internal health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

