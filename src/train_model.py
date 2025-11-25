"""
Навчання моделі для розпізнавання голосових команд
Повний цикл: завантаження даних → навчання → збереження моделі
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import os
import sys

# Додаємо шлях до наших модулів
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simple_data_loader import create_simple_data_loaders
from src.model import SimpleCNN, EvenSimplerCNN

class Trainer:
    """
    Клас для навчання моделі
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Функція втрат для класифікації
        self.criterion = nn.CrossEntropyLoss()
        
        # Оптимізатор Adam (адаптивний градієнтний спуск)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.001,           # Швидкість навчання
            weight_decay=1e-4   # Регуляризація (запобігає перенавчанню)
        )
        
        # Для зберігання статистик
        self.train_losses = []
        self.train_accuracies = []
        
        print(f"🎯 Тренер створено!")
        print(f"   📱 Пристрій: {device}")
        print(f"   💥 Функція втрат: CrossEntropyLoss")
        print(f"   🏃 Оптимізатор: Adam (lr=0.001)")
    
    def train_epoch(self, train_loader):
        """Навчання на одній епосі (повний прохід по всіх даних)"""
        
        self.model.train()  # Режим навчання
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Переносимо дані на пристрій (CPU/GPU)
            data, target = data.to(self.device), target.to(self.device)
            
            # Обнуляємо градієнти
            self.optimizer.zero_grad()
            
            # Прямий прохід: дані → модель → прогнози
            output = self.model(data)
            
            # Обчислюємо втрати (наскільки неправильні прогнози)
            loss = self.criterion(output, target)
            
            # Зворотний прохід: втрати → градієнти
            loss.backward()
            
            # Оновлюємо параметри моделі
            self.optimizer.step()
            
            # Статистики
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Показуємо прогрес кожні 100 батчів
            if batch_idx % 100 == 0:
                print(f'   Батч {batch_idx}/{len(train_loader)}: '
                      f'Втрати={loss.item():.4f}, '
                      f'Точність={100.*correct/total:.1f}%')
        
        # Середні статистики за епоху
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """Оцінка моделі на тестових даних"""
        
        self.model.eval()  # Режим оцінки (вимикаємо dropout тощо)
        
        test_loss = 0
        correct = 0
        total = 0
        
        # Класи для детальної статистики
        classes = ['yes', 'no', 'up', 'down']
        class_correct = [0] * 4
        class_total = [0] * 4
        
        with torch.no_grad():  # Не обчислюємо градієнти (економимо пам'ять)
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Статистики по класах
                for i in range(len(target)):
                    label = target[i].item()
                    class_total[label] += 1
                    if predicted[i] == target[i]:
                        class_correct[label] += 1
        
        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        # Показуємо детальну статистику
        print(f"\n📊 Детальна статистика по класах:")
        for i, class_name in enumerate(classes):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                print(f"   {class_name}: {class_acc:.1f}% ({class_correct[i]}/{class_total[i]})")
        
        return avg_loss, accuracy

def train_model(model_type='simple', epochs=3, batch_size=32, save_model=True):
    """
    Основна функція навчання
    
    Args:
        model_type: 'simple' або 'even_simpler'
        epochs: кількість епох навчання
        batch_size: розмір батча
        save_model: чи зберігати модель після навчання
    """
    
    print("🚀 Починаємо навчання моделі!")
    print("=" * 50)
    
    # Визначаємо пристрій (CPU бо у нас немає GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Використовуємо пристрій: {device}")
    
    # Завантажуємо дані
    print("\n📂 Завантажуємо дані...")
    train_loader, test_loader = create_simple_data_loaders(batch_size=batch_size)
    
    if train_loader is None:
        print("❌ Помилка завантаження даних!")
        return None
    
    # Створюємо модель
    print(f"\n🏗️ Створюємо модель: {model_type}")
    if model_type == 'simple':
        model = SimpleCNN(num_classes=4)
    else:
        model = EvenSimplerCNN(num_classes=4)
    
    # Створюємо тренер
    trainer = Trainer(model, device)
    
    # Оцінка до навчання
    print(f"\n🧪 Тестуємо модель ДО навчання...")
    initial_loss, initial_acc = trainer.evaluate(test_loader)
    print(f"Початкова точність: {initial_acc:.1f}% (випадкова = 25%)")
    
    # Навчання
    print(f"\n🎓 Починаємо навчання на {epochs} епох...")
    start_time = time.time()
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        print(f"\n--- Епоха {epoch + 1}/{epochs} ---")
        
        # Навчаємо одну епоху
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Оцінюємо на тестових даних
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        print(f"📈 Результати епохи {epoch + 1}:")
        print(f"   Тренувальна втрата: {train_loss:.4f}")
        print(f"   Тренувальна точність: {train_acc:.1f}%")
        print(f"   Тестова втрата: {test_loss:.4f}")
        print(f"   Тестова точність: {test_acc:.1f}%")
        
        # Зберігаємо найкращу модель
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            if save_model:
                os.makedirs('./models', exist_ok=True)
                torch.save(model.state_dict(), f'./models/best_model_{model_type}.pth')
                print(f"💾 Збережено кращу модель (точність: {test_acc:.1f}%)")
    
    training_time = time.time() - start_time
    
    # Підсумки
    print(f"\n🎉 Навчання завершено!")
    print("=" * 50)
    print(f"⏱️ Час навчання: {training_time:.1f} секунд")
    print(f"🎯 Найкраща точність: {best_accuracy:.1f}%")
    print(f"📈 Покращення: {best_accuracy - initial_acc:.1f}%")
    
    # Вимірюємо швидкість інференсу
    print(f"\n⚡ Тестуємо швидкість інференсу...")
    model.eval()
    with torch.no_grad():
        # Беремо один батч для тесту
        test_data, _ = next(iter(test_loader))
        test_data = test_data.to(device)
        
        # Вимірюємо час на 100 прогнозів
        start_time = time.time()
        for _ in range(100):
            _ = model(test_data[:1])  # Один зразок
        inference_time = (time.time() - start_time) * 1000 / 100  # мс на зразок
    
    print(f"🔥 Latency: {inference_time:.1f} мс на зразок")
    
    return model, best_accuracy, inference_time

if __name__ == "__main__":
    import argparse
    
    # Парсинг аргументів командного рядка
    parser = argparse.ArgumentParser(description='Train Speech Commands Model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model-type', type=str, default='simple', choices=['simple', 'even_simpler'], help='Model architecture')
    args = parser.parse_args()
    
    print("🎓 Скрипт навчання Speech Commands моделі")
    print("=" * 50)
    
    # Налаштування навчання з аргументів
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MODEL_TYPE = args.model_type
    
    print(f"⚙️ Налаштування:")
    print(f"   Епохи: {EPOCHS}")
    print(f"   Розмір батча: {BATCH_SIZE}")
    print(f"   Тип моделі: {MODEL_TYPE}")
    
    # Запускаємо навчання
    try:
        model, accuracy, latency = train_model(
            model_type=MODEL_TYPE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            save_model=True
        )
        
        if model is not None:
            print(f"\n✅ Успішно навчено модель!")
            print(f"📊 Фінальні метрики:")
            print(f"   🎯 Accuracy: {accuracy:.1f}%")
            print(f"   ⚡ Latency: {latency:.1f} мс")
            
            # Розмір моделі
            model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024  # KB
            print(f"   💾 Розмір моделі: {model_size:.1f} KB")
            
            print(f"\n🎊 Модель готова до використання!")
        
    except Exception as e:
        print(f"❌ Помилка під час навчання: {e}")
        import traceback
        traceback.print_exc()