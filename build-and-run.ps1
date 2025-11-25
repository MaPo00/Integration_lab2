# Скрипт для побудови та запуску Docker контейнера
# Speech Commands Recognition API

Write-Host "🐳 Docker Build and Deploy Script for Speech Commands API" -ForegroundColor Cyan
Write-Host "=" * 60

# Перевірка Docker
Write-Host "`n🔍 Перевіряємо Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker знайдено: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker не встановлений!" -ForegroundColor Red
    Write-Host "📋 Інструкції для встановлення:" -ForegroundColor Yellow
    Write-Host "1. Завантажити Docker Desktop: https://www.docker.com/products/docker-desktop/"
    Write-Host "2. Встановити та перезавантажити систему"
    Write-Host "3. Запустити Docker Desktop"
    exit 1
}

# Параметри
$IMAGE_NAME = "speech-commands-api"
$CONTAINER_NAME = "speech-api"
$LOCAL_PORT = 8000
$CONTAINER_PORT = 5000

Write-Host "`n🏗️ Параметри збірки:" -ForegroundColor Yellow
Write-Host "   Образ: $IMAGE_NAME"
Write-Host "   Контейнер: $CONTAINER_NAME"
Write-Host "   Порт: $LOCAL_PORT -> $CONTAINER_PORT"

# Зупинка та видалення існуючого контейнера
Write-Host "`n🛑 Очищення існуючих контейнерів..." -ForegroundColor Yellow
docker stop $CONTAINER_NAME 2>$null
docker rm $CONTAINER_NAME 2>$null

# Побудова образу
Write-Host "`n🔨 Побудова Docker образу..." -ForegroundColor Yellow
$buildStart = Get-Date
docker build -t "${IMAGE_NAME}:latest" .

if ($LASTEXITCODE -eq 0) {
    $buildEnd = Get-Date
    $buildTime = ($buildEnd - $buildStart).TotalSeconds
    Write-Host "✅ Образ побудовано успішно за $([math]::Round($buildTime, 1)) секунд" -ForegroundColor Green
} else {
    Write-Host "❌ Помилка побудови образу!" -ForegroundColor Red
    exit 1
}

# Інформація про образ
Write-Host "`n📊 Інформація про образ:" -ForegroundColor Yellow
docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Запуск контейнера
Write-Host "`n🚀 Запуск контейнера..." -ForegroundColor Yellow
docker run -d `
    -p "${LOCAL_PORT}:${CONTAINER_PORT}" `
    --name $CONTAINER_NAME `
    --restart unless-stopped `
    --memory="1g" `
    --cpus="1.0" `
    "${IMAGE_NAME}:latest"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Контейнер запущено успішно!" -ForegroundColor Green
} else {
    Write-Host "❌ Помилка запуску контейнера!" -ForegroundColor Red
    exit 1
}

# Очікування запуску
Write-Host "`n⏳ Очікування запуску сервісу..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Перевірка стану
Write-Host "`n🔍 Перевіряємо стан контейнера..." -ForegroundColor Yellow
docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Тестування API
Write-Host "`n🧪 Тестуємо API..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:$LOCAL_PORT/health" -TimeoutSec 10
    Write-Host "✅ API відповідає:" -ForegroundColor Green
    Write-Host "   Статус: $($response.status)" -ForegroundColor White
    Write-Host "   Модель завантажена: $($response.model_loaded)" -ForegroundColor White
} catch {
    Write-Host "⚠️ API поки не доступний, перевірте логи:" -ForegroundColor Orange
    Write-Host "   docker logs $CONTAINER_NAME" -ForegroundColor Gray
}

# Корисні команди
Write-Host "`n📋 Корисні команди:" -ForegroundColor Cyan
Write-Host "   🌐 Відкрити API:     http://localhost:$LOCAL_PORT"
Write-Host "   📝 Переглянути логи: docker logs $CONTAINER_NAME"
Write-Host "   📊 Статистика:       docker stats $CONTAINER_NAME"
Write-Host "   🛑 Зупинити:         docker stop $CONTAINER_NAME"
Write-Host "   🗑️ Видалити:         docker rm $CONTAINER_NAME"

Write-Host "`n🎉 Розгортання завершено!" -ForegroundColor Green
Write-Host "🌐 API доступний на: http://localhost:$LOCAL_PORT" -ForegroundColor Yellow