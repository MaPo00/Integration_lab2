"""
Інференс скрипт для взаємодії з Flask API
Можна тестувати з консолі або аудіо файлами
"""

import requests
import json
import time
import os
import sys
from pathlib import Path

class SpeechCommandsClient:
    """Клієнт для роботи з Speech Commands API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.classes = ['yes', 'no', 'up', 'down']
        
    def check_health(self):
        """Перевіряємо стан API"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "connection_failed"}
    
    def get_api_info(self):
        """Отримуємо інформацію про API"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_from_text(self, text):
        """Тестовий прогноз з тексту"""
        try:
            data = {"text": text}
            response = requests.post(
                f"{self.base_url}/predict_text",
                json=data,
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_from_audio_file(self, audio_path):
        """Прогноз з аудіо файлу"""
        try:
            if not os.path.exists(audio_path):
                return {"error": f"Файл не знайдено: {audio_path}"}
            
            with open(audio_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                response = requests.post(
                    f"{self.base_url}/predict",
                    files=files,
                    timeout=30
                )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def print_prediction_result(result, input_info=""):
    """Красиво виводимо результат прогнозу"""
    print(f"\n🎯 РЕЗУЛЬТАТ РОЗПІЗНАВАННЯ {input_info}")
    print("-" * 40)
    
    if "error" in result:
        print(f"❌ Помилка: {result['error']}")
        return
    
    if result.get("status") != "success" and "simulation" not in result.get("status", ""):
        print(f"⚠️ Статус: {result.get('status', 'unknown')}")
        return
    
    # Основний результат
    predicted = result.get("predicted_class", "unknown")
    confidence = result.get("confidence", 0) * 100
    latency = result.get("inference_time_ms", 0)
    
    print(f"🏆 Передбачена команда: {predicted.upper()}")
    print(f"🎯 Впевненість: {confidence:.1f}%")
    print(f"⚡ Час обробки: {latency:.1f} мс")
    
    # Усі ймовірності
    if "all_probabilities" in result:
        print(f"\n📊 Детальні ймовірності:")
        probs = result["all_probabilities"]
        
        # Сортуємо по ймовірності
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, prob in sorted_probs:
            percentage = prob * 100
            bar_length = int(percentage / 5)  # Масштаб для візуалізації
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            marker = "👑" if class_name == predicted else "  "
            print(f"   {marker} {class_name:>4}: {percentage:5.1f}% [{bar}]")

def interactive_text_mode(client):
    """Інтерактивний режим з текстовим вводом"""
    print("\n💬 РЕЖИМ ТЕКСТОВОГО ТЕСТУВАННЯ")
    print("=" * 40)
    print("Введіть команду або текст що містить команду")
    print("Підтримувані команди: yes, no, up, down")
    print("Введіть 'exit' для виходу\n")
    
    while True:
        try:
            text = input("🎤 Ваш текст: ").strip()
            
            if text.lower() in ['exit', 'quit', 'вихід']:
                print("👋 До побачення!")
                break
                
            if not text:
                print("⚠️ Порожній ввід, спробуйте ще раз")
                continue
            
            # Відправляємо запит
            print("🔄 Обробляємо...")
            result = client.predict_from_text(text)
            
            # Виводимо результат
            print_prediction_result(result, f"для тексту: '{text}'")
            
        except KeyboardInterrupt:
            print("\n👋 Зупинено користувачем")
            break
        except Exception as e:
            print(f"❌ Помилка: {e}")

def test_audio_files(client, audio_dir="./test_audio"):
    """Тестування з аудіо файлами"""
    print(f"\n🎵 РЕЖИМ ТЕСТУВАННЯ АУДІО ФАЙЛІВ")
    print("=" * 40)
    
    if not os.path.exists(audio_dir):
        print(f"⚠️ Папка {audio_dir} не існує")
        print("💡 Створіть папку і покладіть туди .wav файли для тестування")
        return
    
    # Шукаємо аудіо файли
    audio_extensions = ['.wav', '.mp3', '.flac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(audio_dir).glob(f"*{ext}"))
    
    if not audio_files:
        print(f"⚠️ Не знайдено аудіо файлів в {audio_dir}")
        print(f"💡 Підтримувані формати: {', '.join(audio_extensions)}")
        return
    
    print(f"📁 Знайдено {len(audio_files)} аудіо файл(ів):")
    for i, file_path in enumerate(audio_files, 1):
        print(f"   {i}. {file_path.name}")
    
    # Тестуємо кожен файл
    for file_path in audio_files:
        print(f"\n🎵 Тестуємо: {file_path.name}")
        result = client.predict_from_audio_file(str(file_path))
        print_prediction_result(result, f"для файлу: {file_path.name}")

def benchmark_api(client, num_requests=10):
    """Тестуємо продуктивність API"""
    print(f"\n⚡ БЕНЧМАРК API ({num_requests} запитів)")
    print("=" * 40)
    
    test_texts = ["yes please", "no thank you", "go up", "go down"]
    times = []
    
    for i in range(num_requests):
        text = test_texts[i % len(test_texts)]
        
        start_time = time.time()
        result = client.predict_from_text(text)
        end_time = time.time()
        
        request_time = (end_time - start_time) * 1000  # мс
        times.append(request_time)
        
        if "error" not in result:
            inference_time = result.get("inference_time_ms", 0)
            print(f"Запит {i+1:2d}: {request_time:6.1f}мс загально, {inference_time:5.1f}мс інференс")
        else:
            print(f"Запит {i+1:2d}: ПОМИЛКА - {result['error']}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Статистики:")
        print(f"   Середній час: {avg_time:.1f} мс")
        print(f"   Мін/Макс: {min_time:.1f} / {max_time:.1f} мс")
        print(f"   Пропускна здатність: ~{1000/avg_time:.1f} запитів/сек")

def main():
    """Головна функція"""
    print("🎤 Speech Commands API Client")
    print("=" * 50)
    
    # Створюємо клієнт
    client = SpeechCommandsClient()
    
    # Перевіряємо з'єднання
    print("🔗 Перевіряємо з'єднання з API...")
    health = client.check_health()
    
    if "error" in health:
        print(f"❌ Не можу підключитись до API: {health['error']}")
        print("💡 Переконайтесь що Flask сервер запущено на http://localhost:5000")
        return
    
    print(f"✅ API доступний! Статус: {health.get('status', 'unknown')}")
    
    # Отримуємо інформацію про API
    api_info = client.get_api_info()
    if "classes" in api_info:
        print(f"🎯 Підтримувані команди: {', '.join(api_info['classes'])}")
    
    # Головне меню
    while True:
        print(f"\n🎛️ ГОЛОВНЕ МЕНЮ:")
        print("1. 💬 Текстове тестування (інтерактивно)")
        print("2. 🎵 Тестування аудіо файлів")  
        print("3. ⚡ Бенчмарк продуктивності")
        print("4. ℹ️ Інформація про API")
        print("5. 🚪 Вихід")
        
        try:
            choice = input("\n➡️ Ваш вибір (1-5): ").strip()
            
            if choice == "1":
                interactive_text_mode(client)
            elif choice == "2":
                test_audio_files(client)
            elif choice == "3":
                benchmark_api(client)
            elif choice == "4":
                info = client.get_api_info()
                print(f"\n📋 Інформація про API:")
                print(json.dumps(info, indent=2, ensure_ascii=False))
            elif choice == "5":
                print("👋 До побачення!")
                break
            else:
                print("⚠️ Невірний вибір, спробуйте ще раз")
                
        except KeyboardInterrupt:
            print("\n👋 Зупинено користувачем")
            break
        except Exception as e:
            print(f"❌ Помилка: {e}")

if __name__ == "__main__":
    main()