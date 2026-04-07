# Use official Python 3.11 slim runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for fast dependency management (as root)
RUN pip install --no-cache-dir uv

# Copy package metadata AND core engine files
COPY pyproject.toml uv.lock requirements.txt ./
COPY env.py models.py tasks.py ./
COPY server/ ./server/

# Install the environment package into the system (as root)
RUN uv pip install --system --no-cache .

# NOW create and switch to a non-root user for security (Hugging Face requirement)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Safely copy any remaining documentation and frontend files
COPY --chown=user . .

# Expose port 7860 for the dashboard
EXPOSE 7860

# Run the FastAPI server using the standardized OpenEnv v0.2.0 entry point
CMD ["openenv-server"]
