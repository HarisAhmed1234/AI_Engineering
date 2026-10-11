FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHECKPOINT_PATH=/app/checkpoints/minigpt_final.pt

# requirements-api.txt selects a CPU-only PyTorch wheel, separate from the
# CUDA training environment used on the development laptop.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY src ./src
COPY checkpoints/.gitkeep ./checkpoints/.gitkeep

EXPOSE 10000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]

