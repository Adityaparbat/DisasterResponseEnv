FROM python:3.11-slim

WORKDIR /app

# Install dependencies (Caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy core engine
COPY models.py .
COPY tasks.py .
COPY env.py .
COPY app.py .
COPY index.html .

# Copy evaluation & training logic (Submission-Ready)
COPY inference.py .
COPY gym_wrapper.py .
# COPY train.py .
# COPY evaluate_rl.py .

# Copy metadata
COPY openenv.yaml .
COPY README.md .
COPY walkthrough.md .

EXPOSE 7860

# Run with uvicorn for production-grade stability
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
