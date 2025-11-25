
Розпізнавання голосових команд (yes, no, up, down) використовуючи PyTorch CNN та Flask API.

**Фінальні результати: 90.8% точність, 0.3мс латентність** 

---

## 📂 Структура проекту
```
lab1my/
├── src/                           # Вихідний код
│   ├── train_model.py            # Навчання моделі 
│   ├── test_model.py             # Тестування та метрики
│   ├── api.py                    # Flask REST API
│   ├── inference_client.py       # Клієнт для тестування API
│   ├── model.py                  # Архітектура CNN
│   ├── simple_data_loader.py     # Завантаження даних
│   └── preprocessing.py          # Обробка аудіо
├── data/                         # Google Speech Commands dataset
├── models/                       # Навчені моделі
│   └── best_model_simple.pth     # Найкраща модель (90.8%)
├── training_results.md           # Детальні результати 
├── requirements.txt              # Залежності Python
└── README.md                     # Цей файл
```
