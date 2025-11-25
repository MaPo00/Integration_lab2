"""
Попередня обробка аудіо даних для Speech Commands
Перетворюємо звук → спектрограми → тензори для навчання
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Додаємо батьківську директорію до шляху
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import SubsetSC

class AudioPreprocessor:
    """Клас для обробки аудіо даних"""
    
    def __init__(self, sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512):
        """
        Ініціалізація параметрів обробки
        
        Args:
            sample_rate: Частота дискретизації (16kHz стандарт для мови)
            n_mels: Кількість мел-фільтрів (чим більше, тим детальніше)
            n_fft: Розмір вікна FFT 
            hop_length: Крок між вікнами
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Створюємо трансформацію: аудіо → мел-спектрограма
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0  # Енергетична спектрограма
        )
        
        # Логарифмічне масштабування (людський слух логарифмічний)
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
        
        print(f"🔧 Налаштовано аудіо обробник:")
        print(f"   📊 Частота: {sample_rate} Hz")
        print(f"   🎵 Мел-фільтри: {n_mels}")
        
    def __call__(self, waveform):
        """
        Основна функція обробки: звук → спектрограма
        
        Args:
            waveform: Сирий аудіо сигнал [1, samples]
            
        Returns:
            spectrogram: Мел-спектрограма [1, n_mels, time_frames]
        """
        # Нормалізуємо довжину (всі записи мають бути однакової довжини)
        waveform = self._normalize_length(waveform)
        
        # Аудіо → мел-спектрограма
        mel_spec = self.mel_spectrogram(waveform)
        
        # Лінійна → логарифмічна шкала (як чує людина)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Нормалізація в діапазон [0, 1]
        normalized_spec = self._normalize_spectrogram(log_mel_spec)
        
        return normalized_spec
    
    def _normalize_length(self, waveform, target_length=16000):
        """Нормалізуємо довжину аудіо до 1 секунди (16000 сємплів)"""
        current_length = waveform.shape[-1]
        
        if current_length > target_length:
            # Обрізаємо якщо довше
            waveform = waveform[:, :target_length]
        elif current_length < target_length:
            # Доповнюємо нулями якщо коротше
            pad_length = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
        return waveform
    
    def _normalize_spectrogram(self, spectrogram):
        """Нормалізуємо спектрограму до діапазону [0, 1]"""
        # Min-max нормалізація
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        
        if spec_max > spec_min:
            normalized = (spectrogram - spec_min) / (spec_max - spec_min)
        else:
            normalized = spectrogram
            
        return normalized

class SpeechCommandsDataset(Dataset):
    """
    PyTorch Dataset для Speech Commands
    Автоматично застосовує preprocessing до кожного зразка
    """
    
    def __init__(self, subset="training", transform=None):
        """
        Args:
            subset: "training", "validation", або "testing"  
            transform: Функція обробки аудіо (AudioPreprocessor)
        """
        self.dataset = SubsetSC(subset=subset)
        self.transform = transform
        
        # Створюємо мапінг: назва класу → число
        self.classes = ["yes", "no", "up", "down"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"📦 Створено {subset} dataset:")
        print(f"   📊 Зразків: {len(self.dataset)}")
        print(f"   🏷️ Класи: {self.classes}")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Отримуємо один зразок: (спектрограма, мітка)"""
        # Завантажуємо сирі дані
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]
        
        # Застосовуємо обробку якщо є
        if self.transform:
            waveform = self.transform(waveform)
        
        # Перетворюємо мітку в число
        label_idx = self.class_to_idx[label]
        
        return waveform, label_idx

def create_data_loaders(batch_size=32, num_workers=0):
    """
    Створює DataLoader'и для тренування і тестування
    
    Args:
        batch_size: Розмір батча (скільки зразків обробляємо одночасно)
        num_workers: Кількість процесів для завантаження даних
        
    Returns:
        train_loader, test_loader: PyTorch DataLoader'и
    """
    print("🔄 Створюємо обробник даних...")
    
    # Створюємо обробник аудіо
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=64,  # Достатньо для простої моделі
        n_fft=1024,
        hop_length=512
    )
    
    # Створюємо datasets
    print("\n📂 Завантажуємо datasets...")
    train_dataset = SpeechCommandsDataset("training", transform=preprocessor)
    test_dataset = SpeechCommandsDataset("testing", transform=preprocessor)
    
    # Створюємо data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Перемішуємо дані для кращого навчання
        num_workers=num_workers,
        pin_memory=True  # Прискорює на GPU
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Тестові дані не перемішуємо
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✅ DataLoader'и створено:")
    print(f"   🚂 Train batches: {len(train_loader)}")
    print(f"   🧪 Test batches: {len(test_loader)}")
    print(f"   📦 Batch size: {batch_size}")
    
    return train_loader, test_loader

def visualize_preprocessing():
    """Показуємо як виглядає обробка аудіо → спектрограма"""
    print("\n🎨 Створюємо візуалізацію обробки даних...")
    
    # Завантажуємо один зразок
    dataset = SubsetSC("training")
    waveform, sample_rate, label, _, _ = dataset[0]
    
    # Створюємо обробник
    preprocessor = AudioPreprocessor()
    
    # Обробляємо аудіо
    spectrogram = preprocessor(waveform)
    
    # Створюємо графік
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Сирий аудіо сигнал
    time_axis = np.linspace(0, waveform.shape[1] / sample_rate, waveform.shape[1])
    ax1.plot(time_axis, waveform.squeeze().numpy())
    ax1.set_title(f'Сирий аудіо сигнал: "{label}"')
    ax1.set_xlabel('Час (сек)')
    ax1.set_ylabel('Амплітуда')
    ax1.grid(True)
    
    # Спектрограма
    im = ax2.imshow(
        spectrogram.squeeze().numpy(), 
        aspect='auto', 
        origin='lower',
        extent=[0, 1, 0, 64]  # [час_початк, час_кінець, частота_початк, частота_кінець]
    )
    ax2.set_title('Мел-спектрограма (вхід для нейронної мережі)')
    ax2.set_xlabel('Час')
    ax2.set_ylabel('Мел-фільтри')
    plt.colorbar(im, ax=ax2, label='Нормалізована амплітуда')
    
    plt.tight_layout()
    plt.savefig('./data/preprocessing_example.png', dpi=300, bbox_inches='tight')
    print(f"📊 Візуалізація збережена: ./data/preprocessing_example.png")
    plt.show()

if __name__ == "__main__":
    print("🔧 Тестуємо обробку даних...")
    
    try:
        # Створюємо data loaders
        train_loader, test_loader = create_data_loaders(batch_size=4)
        
        # Тестуємо завантаження одного батча
        print("\n🧪 Тестуємо завантаження батча...")
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Батч {batch_idx + 1}:")
            print(f"  📊 Розмір даних: {data.shape}")  # [batch, channels, n_mels, time]
            print(f"  🏷️ Мітки: {target}")
            print(f"  🎯 Класи: {[train_loader.dataset.classes[t] for t in target]}")
            break
            
        # Візуалізація
        visualize_preprocessing()
        
        print("\n✅ Обробка даних налаштована успішно!")
        
    except Exception as e:
        print(f"❌ Помилка: {e}")
        print("Спробуйте спочатку запустити data_loader.py")