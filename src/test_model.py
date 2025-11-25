"""
Тестування навченої моделі
Оцінка метрик: Accuracy, Latency, розмір моделі
"""

import torch
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from collections import Counter

# Додаємо шлях до модулів
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import SimpleCNN, EvenSimplerCNN
from src.simple_data_loader import create_simple_data_loaders

class ModelTester:
    """Клас для детального тестування моделі"""
    
    def __init__(self, model, model_path=None):
        self.model = model
        self.device = torch.device('cpu')
        self.classes = ['yes', 'no', 'up', 'down']
        
        # Завантажуємо ваги якщо є
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Завантажено модель з {model_path}")
        else:
            print(f"⚠️ Використовуємо неначену модель")
            
        self.model.to(self.device)
        self.model.eval()
    
    def calculate_model_size(self):
        """Обчислюємо розмір моделі"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Розмір у байтах (float32 = 4 байти на параметр)
        size_bytes = total_params * 4
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "size_bytes": size_bytes,
            "size_kb": size_kb,
            "size_mb": size_mb
        }
    
    def measure_latency(self, test_loader, num_samples=100):
        """Вимірюємо латентність (час інференсу)"""
        print(f"⏱️ Вимірюємо латентність на {num_samples} зразках...")
        
        # Беремо тестові дані
        all_times = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_data, _ in test_loader:
                batch_data = batch_data.to(self.device)
                
                # Тестуємо кожен зразок у батчі окремо
                for i in range(batch_data.size(0)):
                    if sample_count >= num_samples:
                        break
                        
                    single_sample = batch_data[i:i+1]  # [1, 1, 64, 32]
                    
                    # Вимірюємо час
                    start_time = time.time()
                    output = self.model(single_sample)
                    end_time = time.time()
                    
                    inference_time = (end_time - start_time) * 1000  # мілісекунди
                    all_times.append(inference_time)
                    sample_count += 1
                
                if sample_count >= num_samples:
                    break
        
        # Статистики
        avg_latency = np.mean(all_times)
        median_latency = np.median(all_times)
        min_latency = np.min(all_times)
        max_latency = np.max(all_times)
        std_latency = np.std(all_times)
        
        return {
            "average_ms": avg_latency,
            "median_ms": median_latency,
            "min_ms": min_latency,
            "max_ms": max_latency,
            "std_ms": std_latency,
            "samples_tested": len(all_times)
        }
    
    def evaluate_accuracy(self, test_loader):
        """Детальна оцінка точності"""
        print(f"🎯 Оцінюємо точність на тестовому наборі...")
        
        all_predictions = []
        all_targets = []
        class_correct = [0] * len(self.classes)
        class_total = [0] * len(self.classes)
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                # Загальна статистика
                total_samples += targets.size(0)
                total_correct += (predicted == targets).sum().item()
                
                # Статистика по класах
                for i in range(len(targets)):
                    target_class = targets[i].item()
                    predicted_class = predicted[i].item()
                    
                    class_total[target_class] += 1
                    if target_class == predicted_class:
                        class_correct[target_class] += 1
                    
                    all_predictions.append(predicted_class)
                    all_targets.append(target_class)
        
        # Загальна точність
        overall_accuracy = 100.0 * total_correct / total_samples
        
        # Точність по класах
        class_accuracies = {}
        for i, class_name in enumerate(self.classes):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                class_accuracies[class_name] = {
                    "accuracy": acc,
                    "correct": class_correct[i],
                    "total": class_total[i]
                }
            else:
                class_accuracies[class_name] = {
                    "accuracy": 0.0,
                    "correct": 0,
                    "total": 0
                }
        
        # Матриця плутанини (confusion matrix)
        confusion_matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        for true_label, pred_label in zip(all_targets, all_predictions):
            confusion_matrix[true_label][pred_label] += 1
        
        return {
            "overall_accuracy": overall_accuracy,
            "class_accuracies": class_accuracies,
            "confusion_matrix": confusion_matrix.tolist(),
            "total_samples": total_samples,
            "total_correct": total_correct
        }
    
    def run_full_evaluation(self, test_loader):
        """Повна оцінка моделі"""
        print("🔍 Запускаємо повну оцінку моделі...")
        print("=" * 50)
        
        # 1. Розмір моделі
        size_info = self.calculate_model_size()
        
        # 2. Точність
        accuracy_info = self.evaluate_accuracy(test_loader)
        
        # 3. Латентність
        latency_info = self.measure_latency(test_loader)
        
        # Зводка результатів
        results = {
            "model_size": size_info,
            "accuracy": accuracy_info,
            "latency": latency_info
        }
        
        return results

def print_results(results):
    """Красиво виводимо результати"""
    print("\n📊 РЕЗУЛЬТАТИ ОЦІНКИ МОДЕЛІ")
    print("=" * 50)
    
    # Розмір моделі
    size = results["model_size"]
    print(f"💾 РОЗМІР МОДЕЛІ:")
    print(f"   Параметрів: {size['total_params']:,}")
    print(f"   Розмір: {size['size_kb']:.1f} KB ({size['size_mb']:.2f} MB)")
    
    # Точність
    acc = results["accuracy"]
    print(f"\n🎯 ТОЧНІСТЬ:")
    print(f"   Загальна точність: {acc['overall_accuracy']:.1f}%")
    print(f"   Зразків протестовано: {acc['total_samples']}")
    print(f"   Правильних відповідей: {acc['total_correct']}")
    
    print(f"\n📈 Точність по класах:")
    for class_name, class_info in acc["class_accuracies"].items():
        accuracy = class_info["accuracy"]
        correct = class_info["correct"]
        total = class_info["total"]
        print(f"   {class_name:>4}: {accuracy:5.1f}% ({correct:3d}/{total:3d})")
    
    # Латентність
    lat = results["latency"]
    print(f"\n⚡ ШВИДКІСТЬ (LATENCY):")
    print(f"   Середня: {lat['average_ms']:.2f} мс")
    print(f"   Медіана: {lat['median_ms']:.2f} мс")
    print(f"   Мін/Макс: {lat['min_ms']:.2f} / {lat['max_ms']:.2f} мс")
    
    # Матриця плутанини
    print(f"\n🔀 МАТРИЦЯ ПЛУТАНИНИ:")
    classes = ['yes', 'no', 'up', 'down']
    cm = np.array(acc["confusion_matrix"])
    
    print("      " + "".join(f"{cls:>6}" for cls in classes))
    for i, true_class in enumerate(classes):
        row_str = f"{true_class:>4}: "
        for j in range(len(classes)):
            row_str += f"{cm[i][j]:>6}"
        print(row_str)

def main():
    """Основна функція тестування"""
    print("🧪 Тестування Speech Commands моделі")
    print("=" * 50)
    
    # Параметри
    MODEL_TYPE = 'simple'  # або 'even_simpler'
    MODEL_PATH = f'./models/best_model_{MODEL_TYPE}.pth'
    
    # Завантажуємо тестові дані
    print("📂 Завантажуємо тестові дані...")
    _, test_loader = create_simple_data_loaders(batch_size=32)
    
    if test_loader is None:
        print("❌ Помилка завантаження даних!")
        return
    
    # Створюємо модель
    print(f"🏗️ Створюємо модель: {MODEL_TYPE}")
    if MODEL_TYPE == 'simple':
        model = SimpleCNN(num_classes=4)
    else:
        model = EvenSimplerCNN(num_classes=4)
    
    # Створюємо тестер
    tester = ModelTester(model, MODEL_PATH)
    
    # Запускаємо оцінку
    results = tester.run_full_evaluation(test_loader)
    
    # Виводимо результати
    print_results(results)
    
    # Висновки та рекомендації
    accuracy = results["accuracy"]["overall_accuracy"]
    latency = results["latency"]["average_ms"]
    size_kb = results["model_size"]["size_kb"]
    
    print(f"\n🎊 ВИСНОВКИ:")
    print("=" * 30)
    
    if accuracy >= 70:
        print("✅ Відмінна точність!")
    elif accuracy >= 50:
        print("🟡 Прийнятна точність")
    else:
        print("🔴 Низька точність, потрібне додаткове навчання")
    
    if latency <= 50:
        print("✅ Швидкий інференс!")
    elif latency <= 100:
        print("🟡 Помірна швидкість")
    else:
        print("🔴 Повільний інференс")
        
    if size_kb <= 1000:
        print("✅ Компактна модель!")
    else:
        print("🟡 Велика модель")
    
    print(f"\n🎯 Підсумкові метрики:")
    print(f"   Accuracy: {accuracy:.1f}%")
    print(f"   Latency: {latency:.1f} мс") 
    print(f"   Size: {size_kb:.0f} KB")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print(f"\n✅ Тестування завершено успішно!")
        
    except Exception as e:
        print(f"❌ Помилка під час тестування: {e}")
        import traceback
        traceback.print_exc()