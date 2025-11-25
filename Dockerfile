# ===================================================================
# Multi-stage Dockerfile для Speech Commands Recognition API
# Оптимізований для мінімального розміру та максимальної безпеки
# ===================================================================

# ===== Етап 1: Builder (збірка та встановлення залежностей) =====
FROM python:3.10-slim as builder

# Встановлюємо системні залежності для збірки
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Створюємо віртуальне середовище для ізоляції залежностей
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Встановлюємо Python залежності
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ===== Етап 2: Production (мінімальний runtime контейнер) =====
FROM python:3.10-slim as production

# Встановлюємо тільки runtime залежності (без build tools)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Створюємо non-root користувача для безпеки
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --create-home --shell /bin/bash app

# Копіюємо віртуальне середовище з builder етапу
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Створюємо робочу директорію
WORKDIR /app

# Копіюємо тільки необхідні файли
COPY --chown=app:appgroup src/ ./src/
COPY --chown=app:appgroup models/ ./models/
COPY --chown=app:appgroup requirements.txt ./

# Переключаємося на non-root користувача
USER app

# Відкриваємо порт для Flask
EXPOSE 5000

# Встановлюємо змінні середовища
ENV PYTHONPATH=/app
ENV FLASK_APP=src/api.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check для моніторингу
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Команда запуску з автоматичним перезапуском
CMD ["python", "-u", "src/api.py"]

# Metadata для документації
LABEL maintainer="Yurii <yurii@university.edu>" \
      description="Containerized Speech Commands Recognition API with PyTorch CNN and Flask" \
      version="2.0" \
      dockerfile.stage="production" \
      org.opencontainers.image.title="Speech Commands API" \
      org.opencontainers.image.description="ML inference API for speech command classification (yes/no/up/down)" \
      org.opencontainers.image.vendor="University Integration Project" \
      org.opencontainers.image.licenses="MIT"