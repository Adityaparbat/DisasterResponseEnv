FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy manifest and lock files for optimized caching
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache .

# Copy core engine components (for server logic)
COPY env.py models.py tasks.py index.html ./

# Copy server logic (Required for v0.2.0 compliance)
COPY server/ ./server/

# Copy evaluation & documentation
COPY inference.py gym_wrapper.py README.md walkthrough.md ./

EXPOSE 7860

# Use the standardized OpenEnv server entry point
CMD ["openenv-server"]
