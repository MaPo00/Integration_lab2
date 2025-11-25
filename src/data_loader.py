"""
Завантаження та дослідження Google Speech Commands датасету
"""

import os
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class SubsetSC(SPEECHCOMMANDS):
    """Підклас для вибору конкретних команд"""
    def __init__(self, subset: str = None):
        super().__init__("./data", download=True, subset=subset)
        
        # Обмежуємося 4 класами як у завданні
        self.classes = ["yes", "no", "up", "down"]
        
        # Фільтруємо дані тільки для наших класів
        self._walker = [w for w in self._walker 
                       if w.split(os.path.sep)[-2] in self.classes]
        
        print(f"Завантажено {len(self._walker)} зразків для класів: {self.classes}")

def explore_dataset():
    """Досліджуємо структуру датасету"""
    print("=== Дослідження Speech Commands датасету ===")
    
    # Завантажуємо тренувальний набір
    train_set = SubsetSC(subset="training")
    
    print(f"Загальна кількість тренувальних зразків: {len(train_set)}")
    
    # Аналізуємо розподіл класів
    labels = []
    for i in range(len(train_set)):
        waveform, sample_rate, label, speaker_id, utterance_number = train_set[i]
        labels.append(label)
    
    # Підраховуємо кількість зразків по класах
    class_counts = Counter(labels)
    print("\nРозподіл класів:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} зразків")
    
    # Досліджуємо перший зразок
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
    
    print(f"\nІнформація про перший зразок:")
    print(f"  Форма сигналу: {waveform.shape}")
    print(f"  Частота дискретизації: {sample_rate} Hz")
    print(f"  Тривалість: {waveform.shape[1] / sample_rate:.2f} секунд")
    print(f"  Мітка: {label}")
    print(f"  ID спікера: {speaker_id}")
    
    # Візуалізуємо кілька зразків
    visualize_samples(train_set)
    
    return train_set

def visualize_samples(dataset, num_samples=4):
    """Візуалізуємо кілька аудіо зразків"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for i in range(num_samples):
        waveform, sample_rate, label, _, _ = dataset[i * 100]  # Беремо через 100
        
        # Перетворюємо в numpy для візуалізації
        audio_data = waveform.squeeze().numpy()
        time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        
        axes[i].plot(time, audio_data)
        axes[i].set_title(f'Клас: "{label}"')
        axes[i].set_xlabel('Час (сек)')
        axes[i].set_ylabel('Амплітуда')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('./data/sample_waveforms.png', dpi=300, bbox_inches='tight')
    print(f"\nГрафік збережено: ./data/sample_waveforms.png")
    plt.show()

def check_data_splits():
    """Перевіряємо розподіл на тренувальну та валідаційну вибірки"""
    print("\n=== Перевірка розподілу даних ===")
    
    train_set = SubsetSC(subset="training")
    validation_set = SubsetSC(subset="validation")
    test_set = SubsetSC(subset="testing")
    
    print(f"Тренувальний набір: {len(train_set)} зразків")
    print(f"Валідаційний набір: {len(validation_set)} зразків")  
    print(f"Тестовий набір: {len(test_set)} зразків")
    
    return train_set, validation_set, test_set

if __name__ == "__main__":
    print("Починаємо завантаження та дослідження датасету...")
    
    # Створюємо папку для даних якщо її немає
    os.makedirs("./data", exist_ok=True)
    
    # Досліджуємо датасет
    train_set = explore_dataset()
    
    # Перевіряємо розподіл
    train_set, validation_set, test_set = check_data_splits()
    
    print(f"\n✅ Датасет успішно завантажено та проаналізовано!")
    print(f"📊 Ми будемо працювати з 4 класами: yes, no, up, down")
    print(f"🎯 Готові до наступного кроку - попередньої обробки даних!")