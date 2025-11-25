# 🐳 Docker Команди для проекту Speech Commands API

## 🚀 Швидкий старт

```powershell
# 1. Побудувати образ
docker build -t speech-commands-api:v2 .

# 2. Запустити контейнер
docker run -d --name speech-api -p 8000:5000 speech-commands-api:v2

# 3. Перевірити статус
docker ps

# 4. Тестувати API
curl http://localhost:8000/health
```

## 📊 Моніторинг та діагностика

```powershell
# Переглянути логи
docker logs speech-api

# Статистика ресурсів
docker stats speech-api --no-stream

# Зайти в контейнер
docker exec -it speech-api bash

# Інспектувати контейнер
docker inspect speech-api
```

## 🧹 Управління контейнерами

```powershell
# Зупинити контейнер
docker stop speech-api

# Видалити контейнер
docker rm speech-api

# Видалити образ
docker rmi speech-commands-api:v2

# Очистити всі unused resources
docker system prune -a
```

## 📋 Корисна інформація

### Порти:
- **Host**: 8000
- **Container**: 5000

### Endpoints:
- `GET /` - Інформація про API
- `GET /health` - Health check
- `POST /predict_text` - Тестовий inference

### Файли:
- Модель: `/app/models/best_model_simple.pth`
- Код: `/app/src/`
- Логи: `docker logs speech-api`