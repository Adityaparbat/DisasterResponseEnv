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

# Install requirements (Optimized Caching)
COPY --chown=user requirements.txt pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache .

# Safely copy ONLY the files that were uploaded to the Space
# Ensure proper ownership for the non-root user
COPY --chown=user . .

# Expose port 7860
EXPOSE 7860

# Run the FastAPI server using the standardized OpenEnv v0.2.0 entry point
CMD ["openenv-server"]
