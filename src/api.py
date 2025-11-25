"""
Flask API для розпізнавання голосових команд
REST API endpoints для інференсу моделі
"""

from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import io
import time
import os
import sys
import tempfile

# Додаємо шлях до модулів
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import SimpleCNN, EvenSimplerCNN
from src.simple_data_loader import SimpleAudioPreprocessor

app = Flask(__name__)

# Глобальні змінні для моделі
model = None
preprocessor = None
classes = ['yes', 'no', 'up', 'down']
device = torch.device('cpu')

def load_model(model_path='./models/best_model_simple.pth', model_type='simple'):
    """Завантажуємо навчену модель"""
    global model, preprocessor
    
    print(f"🔄 Завантажуємо модель з {model_path}...")
    
    # Створюємо архітектуру моделі
    if model_type == 'simple':
        model = SimpleCNN(num_classes=4)
    else:
        model = EvenSimplerCNN(num_classes=4)
    
    # Завантажуємо ваги
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Модель завантажено успішно!")
    else:
        print(f"⚠️ Файл моделі не знайдено: {model_path}")
        print("🔄 Використовуємо неначену модель (випадкові ваги)")
        model.eval()
    
    # Створюємо препроцесор
    preprocessor = SimpleAudioPreprocessor()
    
    return True

def predict_audio(audio_data, sample_rate=16000):
    """
    Розпізнавання голосової команди з аудіо
    
    Args:
        audio_data: numpy array з аудіо сигналом
        sample_rate: частота дискретизації
        
    Returns:
        dict з результатами прогнозу
    """
    global model, preprocessor
    
    if model is None or preprocessor is None:
        return {"error": "Модель не завантажена"}
    
    start_time = time.time()
    
    try:
        # Перетворюємо в tensor
        if len(audio_data.shape) == 1:
            audio_tensor = torch.from_numpy(audio_data[np.newaxis, :]).float()
        else:
            audio_tensor = torch.from_numpy(audio_data).float()
        
        # Застосовуємо preprocessing
        processed_audio = preprocessor(audio_tensor)
        
        # Додаємо batch dimension
        if len(processed_audio.shape) == 3:
            processed_audio = processed_audio.unsqueeze(0)
        
        # Прогноз
        with torch.no_grad():
            output = model(processed_audio)
            probabilities = F.softmax(output, dim=1)
            
            # Найкраща класифікація
            _, predicted_idx = torch.max(output, 1)
            predicted_class = classes[predicted_idx.item()]
            confidence = probabilities[0][predicted_idx].item()
        
        # Усі ймовірності
        all_probabilities = {
            classes[i]: float(probabilities[0][i]) 
            for i in range(len(classes))
        }
        
        inference_time = (time.time() - start_time) * 1000  # мілісекунди
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "inference_time_ms": inference_time,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.route('/')
def home():
    """Головна сторінка API"""
    return jsonify({
        "message": "Speech Commands Recognition API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Розпізнавання з аудіо файлу",
            "/predict_text": "POST - Тестовий прогноз з тексту",
            "/health": "GET - Перевірка стану API",
            "/classes": "GET - Список підтримуваних класів"
        },
        "classes": classes,
        "status": "running"
    })

@app.route('/health')
def health():
    """Перевірка стану API"""
    model_loaded = model is not None
    return jsonify({
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "classes": classes,
        "device": str(device)
    })

@app.route('/classes')
def get_classes():
    """Отримати список підтримуваних класів"""
    return jsonify({
        "classes": classes,
        "count": len(classes)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Розпізнавання голосової команди з аудіо файлу
    
    Очікує multipart/form-data з файлом 'audio'
    Підтримує формати: .wav, .mp3, .flac
    """
    
    if 'audio' not in request.files:
        return jsonify({
            "error": "Не знайдено аудіо файл. Використовуйте поле 'audio'",
            "status": "error"
        }), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({
            "error": "Файл не вибрано",
            "status": "error"
        }), 400
    
    try:
        # Читаємо аудіо напряму з пам'яті
        audio_file.seek(0)  # Повертаємося на початок файлу
        audio_data, sample_rate = sf.read(io.BytesIO(audio_file.read()))
        
        # Розпізнаємо
        result = predict_audio(audio_data, sample_rate)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Помилка обробки аудіо: {str(e)}",
            "status": "error"
        }), 500

@app.route('/predict_text', methods=['POST'])
def predict_text():
    """
    Тестовий endpoint для прогнозу з тексту
    Симулює роботу API без реального аудіо
    """
    
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({
            "error": "Потрібен параметр 'text' в JSON",
            "status": "error"
        }), 400
    
    text = data['text'].lower().strip()
    
    # Проста симуляція: якщо текст містить одну з команд
    predicted_class = "unknown"
    confidence = 0.25  # Базова ймовірність
    
    for class_name in classes:
        if class_name in text:
            predicted_class = class_name
            confidence = 0.85 + np.random.random() * 0.1  # 85-95%
            break
    
    # Генеруємо реалістичні ймовірності
    all_probabilities = {}
    remaining_prob = 1.0 - confidence
    
    for i, class_name in enumerate(classes):
        if class_name == predicted_class:
            all_probabilities[class_name] = confidence
        else:
            # Розподіляємо решту ймовірності
            prob = remaining_prob / (len(classes) - 1)
            prob += np.random.random() * 0.05  # Трохи рандомності
            all_probabilities[class_name] = min(prob, remaining_prob)
    
    # Нормалізуємо щоб сума була 1.0
    total = sum(all_probabilities.values())
    all_probabilities = {k: v/total for k, v in all_probabilities.items()}
    
    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "inference_time_ms": np.random.uniform(5, 25),  # Симуляція часу
        "input_text": text,
        "status": "success (simulation)"
    })

@app.errorhandler(404)
def not_found(error):
    """Обробка 404 помилок"""
    return jsonify({
        "error": "Endpoint не знайдено",
        "available_endpoints": ["/", "/health", "/classes", "/predict", "/predict_text"],
        "status": "error"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Обробка внутрішніх помилок"""
    return jsonify({
        "error": "Внутрішня помилка сервера",
        "status": "error"
    }), 500

if __name__ == '__main__':
    print("🚀 Запускаємо Flask API для розпізнавання голосових команд")
    print("=" * 60)
    
    # Завантажуємо модель
    load_model()
    
    print(f"\n📡 API endpoints:")
    print(f"   GET  /           - Інформація про API")
    print(f"   GET  /health     - Стан здоров'я")
    print(f"   GET  /classes    - Підтримувані класи") 
    print(f"   POST /predict    - Розпізнавання аудіо файлу")
    print(f"   POST /predict_text - Тестовий прогноз")
    
    print(f"\n🎯 Підтримувані команди: {', '.join(classes)}")
    
    print(f"\n🔥 Запускаємо сервер на http://localhost:5000")
    print("💡 Натисни Ctrl+C для зупинки")
    
    # Запускаємо Flask сервер
    app.run(
        host='0.0.0.0',     # Доступ з будь-якої IP
        port=5000,          # Порт
        debug=True,         # Режим розробки
        use_reloader=False  # Не перезавантажуємо (щоб не конфліктувати з навчанням)
    )