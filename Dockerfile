# Use official Python 3.11 slim runtime
FROM python:3.11-slim

# Hugging Face Spaces require running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy package metadata AND core engine files (Required for build backend)
COPY --chown=user requirements.txt pyproject.toml uv.lock ./
COPY --chown=user env.py models.py tasks.py ./
COPY --chown=user server/ ./server/

# Build and install the environment package now that source files are present
RUN uv pip install --system --no-cache .

# Safely copy any remaining files (README, walkthrough, etc.)
COPY --chown=user . .

# Expose port 7860
EXPOSE 7860

# Run the FastAPI server using the standardized OpenEnv v0.2.0 entry point
CMD ["openenv-server"]
