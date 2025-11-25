# Скрипт для тестування Docker API vs локального запуску
# Performance comparison script

Write-Host "📊 Performance Comparison: Docker vs Local" -ForegroundColor Cyan
Write-Host "=" * 50

# Параметри тестування
$TEST_ITERATIONS = 10
$DOCKER_URL = "http://localhost:8000"
$LOCAL_URL = "http://localhost:5000"  # Припустимо локальний запуск на 5000

Write-Host "`n🎯 Параметри тестування:" -ForegroundColor Yellow
Write-Host "   Ітерацій: $TEST_ITERATIONS"
Write-Host "   Docker API: $DOCKER_URL"
Write-Host "   Local API: $LOCAL_URL"

# Функція для тестування API
function Test-API {
    param(
        [string]$Url,
        [string]$Name,
        [int]$Iterations
    )
    
    Write-Host "`n🧪 Тестуємо $Name..." -ForegroundColor Yellow
    
    $results = @()
    $errors = 0
    
    for ($i = 1; $i -le $Iterations; $i++) {
        try {
            $start = Get-Date
            
            # Тестуємо health endpoint
            $response = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 5
            
            $end = Get-Date
            $latency = ($end - $start).TotalMilliseconds
            
            $results += $latency
            Write-Host "   Запит $i`: $([math]::Round($latency, 2)) мс" -ForegroundColor Gray
            
        } catch {
            $errors++
            Write-Host "   Запит $i`: ERROR" -ForegroundColor Red
        }
        
        Start-Sleep -Milliseconds 100  # Короткая пауза між запитами
    }
    
    if ($results.Count -gt 0) {
        $avg = ($results | Measure-Object -Average).Average
        $min = ($results | Measure-Object -Minimum).Minimum
        $max = ($results | Measure-Object -Maximum).Maximum
        
        return @{
            Name = $Name
            Url = $Url
            Average = $avg
            Min = $min
            Max = $max
            Errors = $errors
            Success = $results.Count
        }
    } else {
        return @{
            Name = $Name
            Url = $Url
            Average = 0
            Min = 0
            Max = 0
            Errors = $errors
            Success = 0
        }
    }
}

# Перевірка доступності Docker API
Write-Host "`n🔍 Перевіряємо доступність API..." -ForegroundColor Yellow
try {
    $dockerHealth = Invoke-RestMethod -Uri "$DOCKER_URL/health" -TimeoutSec 5
    Write-Host "✅ Docker API доступний" -ForegroundColor Green
    $dockerAvailable = $true
} catch {
    Write-Host "❌ Docker API недоступний" -ForegroundColor Red
    $dockerAvailable = $false
}

try {
    $localHealth = Invoke-RestMethod -Uri "$LOCAL_URL/health" -TimeoutSec 5
    Write-Host "✅ Local API доступний" -ForegroundColor Green
    $localAvailable = $true
} catch {
    Write-Host "❌ Local API недоступний" -ForegroundColor Red
    Write-Host "   💡 Для запуску: python src/api.py" -ForegroundColor Gray
    $localAvailable = $false
}

# Тестування продуктивності
$dockerResults = $null
$localResults = $null

if ($dockerAvailable) {
    $dockerResults = Test-API -Url $DOCKER_URL -Name "Docker Container" -Iterations $TEST_ITERATIONS
}

if ($localAvailable) {
    $localResults = Test-API -Url $LOCAL_URL -Name "Local Python" -Iterations $TEST_ITERATIONS
}

# Результати
Write-Host "`n📊 РЕЗУЛЬТАТИ ТЕСТУВАННЯ:" -ForegroundColor Cyan
Write-Host "=" * 50

if ($dockerResults) {
    Write-Host "`n🐳 Docker Container:" -ForegroundColor Blue
    Write-Host "   Середня латентність: $([math]::Round($dockerResults.Average, 2)) мс" -ForegroundColor White
    Write-Host "   Мін/Макс: $([math]::Round($dockerResults.Min, 2)) / $([math]::Round($dockerResults.Max, 2)) мс" -ForegroundColor White
    Write-Host "   Успішних запитів: $($dockerResults.Success)/$TEST_ITERATIONS" -ForegroundColor White
    Write-Host "   Помилок: $($dockerResults.Errors)" -ForegroundColor White
}

if ($localResults) {
    Write-Host "`n💻 Local Python:" -ForegroundColor Green
    Write-Host "   Середня латентність: $([math]::Round($localResults.Average, 2)) мс" -ForegroundColor White
    Write-Host "   Мін/Макс: $([math]::Round($localResults.Min, 2)) / $([math]::Round($localResults.Max, 2)) мс" -ForegroundColor White
    Write-Host "   Успішних запитів: $($localResults.Success)/$TEST_ITERATIONS" -ForegroundColor White
    Write-Host "   Помилок: $($localResults.Errors)" -ForegroundColor White
}

# Порівняння
if ($dockerResults -and $localResults) {
    Write-Host "`n⚖️ ПОРІВНЯННЯ:" -ForegroundColor Yellow
    
    $dockerOverhead = $dockerResults.Average - $localResults.Average
    $overheadPercent = ($dockerOverhead / $localResults.Average) * 100
    
    Write-Host "   Docker overhead: +$([math]::Round($dockerOverhead, 2)) мс (+$([math]::Round($overheadPercent, 1))%)" -ForegroundColor White
    
    if ($overheadPercent -lt 10) {
        Write-Host "   ✅ Мінімальний overhead - відмінно!" -ForegroundColor Green
    } elseif ($overheadPercent -lt 25) {
        Write-Host "   🟡 Помірний overhead - прийнятно" -ForegroundColor Yellow
    } else {
        Write-Host "   ⚠️ Значний overhead - потребує оптимізації" -ForegroundColor Orange
    }
}

# Інформація про розміри
Write-Host "`n📦 РОЗМІР АРТЕФАКТІВ:" -ForegroundColor Cyan

# Docker образ
try {
    $dockerSize = docker images speech-commands-api --format "{{.Size}}" 2>$null
    if ($dockerSize) {
        Write-Host "   🐳 Docker образ: $dockerSize" -ForegroundColor White
    }
} catch {
    Write-Host "   🐳 Docker образ: Не знайдено" -ForegroundColor Gray
}

# Локальні файли
try {
    $modelSize = (Get-Item "models/best_model_simple.pth" -ErrorAction SilentlyContinue).Length
    if ($modelSize) {
        $modelSizeMB = [math]::Round($modelSize / 1MB, 2)
        Write-Host "   🤖 Модель: $modelSizeMB MB" -ForegroundColor White
    }
} catch {
    Write-Host "   🤖 Модель: Не знайдено" -ForegroundColor Gray
}

# Підсумок
Write-Host "`n📋 РЕКОМЕНДАЦІЇ:" -ForegroundColor Cyan

if ($dockerAvailable -and $localAvailable) {
    Write-Host "✅ Docker контейнеризація успішна" -ForegroundColor Green
    Write-Host "✅ API функціонує в обох режимах" -ForegroundColor Green
    
    if ($overheadPercent -lt 15) {
        Write-Host "🎯 Контейнер готовий для production" -ForegroundColor Green
    } else {
        Write-Host "🔧 Рекомендується оптимізація контейнера" -ForegroundColor Orange
    }
} elseif ($dockerAvailable) {
    Write-Host "✅ Docker контейнер працює" -ForegroundColor Green
    Write-Host "⚠️ Локальний запуск для порівняння недоступний" -ForegroundColor Orange
} else {
    Write-Host "❌ Потрібно налаштувати Docker середовище" -ForegroundColor Red
}

Write-Host "`n🎉 Тестування завершено!" -ForegroundColor Cyan