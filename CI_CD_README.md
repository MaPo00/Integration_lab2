# 🚀 CI/CD Pipeline Documentation

## 📋 Огляд

Автоматизований CI/CD пайплайн для тренування, тестування та деплою ML моделі розпізнавання голосових команд.

---

## 🔄 Workflow Jobs

### 1️⃣ **Train Model** (train-model)
Тренує ML модель у Docker контейнері:
- Будує training Docker image
- Запускає навчання з параметрами
- Зберігає навчену модель як артефакт
- Генерує метрики навчання

### 2️⃣ **Test Model** (test-model)
Перевіряє якість навченої моделі:
- Завантажує модель з артефактів
- Запускає тести на стабільних зразках
- Валідує точність моделі
- Блокує деплой при невдалих тестах

### 3️⃣ **Build & Push Inference** (build-and-push-inference)
Створює та публікує production образ:
- Будує inference Docker image
- Пушить до GitHub Container Registry (GHCR)
- Генерує deployment звіт
- Тегує образи за версією та SHA

### 4️⃣ **Benchmark** (benchmark)
Вимірює performance (тільки на main):
- Запускає inference контейнер
- Вимірює latency
- Зберігає результати як артефакти

---

## ⚡ Тригери

### 🔵 **Push до main**
```yaml
push:
  branches:
    - main
```
Запускає повний пайплайн: train → test → build → benchmark

### 🟢 **Pull Request**
```yaml
pull_request:
  branches:
    - main
```
Запускає train → test (без деплою)

### 🟡 **Manual Run**
```yaml
workflow_dispatch:
  inputs:
    epochs:
      description: 'Number of training epochs'
      default: '3'
```
Ручний запуск через GitHub UI з можливістю вказати кількість епох

---

## 📦 Артефакти

| Артефакт | Опис | Retention |
|----------|------|-----------|
| `trained-model` | Навчена модель (*.pth) | 30 днів |
| `training-metrics` | JSON з метриками + логи | 30 днів |
| `deployment-report` | Markdown звіт про деплой | 90 днів |
| `benchmark-results` | Результати latency тестів | 30 днів |

---

## 🐳 Docker Images

### Registry
**GitHub Container Registry (GHCR)**  
`ghcr.io/mapo00/integration_lab2`

### Tags
- `latest` - остання версія з main
- `sha-<commit>` - конкретний коміт
- `pr-<number>` - pull request preview

### Pull Image
```bash
docker pull ghcr.io/mapo00/integration_lab2:latest
```

### Run Inference
```bash
docker run -p 8000:5000 ghcr.io/mapo00/integration_lab2:latest
```

---

## 🔐 Secrets та Permissions

### Required Secrets
Створені автоматично GitHub:
- `GITHUB_TOKEN` - для push до GHCR

### Required Permissions
```yaml
permissions:
  contents: read
  packages: write
```

---

## 🛡️ Branch Protection Rules

### Main Branch Protection
- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- ✅ Block merge if CI fails
- ✅ Require pull request reviews

### Setup
1. Repository Settings → Branches
2. Add rule for `main`
3. Enable "Require status checks"
4. Select: `train-model`, `test-model`

---

## 📊 Monitoring

### View Workflow Runs
```
https://github.com/MaPo00/Integration_lab2/actions
```

### Download Artifacts
```bash
# Using GitHub CLI
gh run download <run-id> -n trained-model
```

### Check Image in Registry
```
https://github.com/MaPo00?tab=packages
```

---

## 🧪 Local Testing

### Test Docker Build
```bash
# Build training image
docker build --target builder -t training-image:local .

# Build inference image
docker build --target production -t inference-image:local .
```

### Test Training
```bash
docker run -v $(pwd)/models:/app/models \
  training-image:local python src/train_model.py --epochs 1
```

### Test Inference
```bash
docker run -p 8000:5000 inference-image:local
curl http://localhost:8000/health
```

---

## 🔧 Troubleshooting

### Pipeline Failed?
1. Check logs in Actions tab
2. Verify Docker build succeeds locally
3. Check model artifacts uploaded
4. Validate test_model.py works

### Can't Push to GHCR?
1. Verify packages write permission
2. Check GITHUB_TOKEN has correct scope
3. Ensure image name is lowercase

### Model Test Failed?
1. Check model file exists in artifacts
2. Verify test data available
3. Review accuracy threshold

---

## 📚 Best Practices

### ✅ DO
- Keep epochs low (1-3) for CI testing
- Cache Docker layers for speed
- Use artifacts for model transfer
- Log all important metrics
- Tag images with commit SHA

### ❌ DON'T
- Train on full dataset in CI
- Store secrets in code
- Skip testing before deploy
- Use latest tag in production
- Commit large binary files

---

## 🎯 Next Steps

1. Set up branch protection rules
2. Add more comprehensive tests
3. Implement model versioning
4. Add deployment to cloud
5. Set up monitoring alerts

---

*Generated for Integration Lab 3 - CI/CD Automation*