"""
Альтернативний data loader що використовує soundfile
"""

import os
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from collections import Counter

class SimpleSpeechCommandsDataset(Dataset):
    """
    Простий dataset що читає файли напряму з папок
    """
    
    def __init__(self, root_dir, classes=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes or ["yes", "no", "up", "down"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Шукаємо всі аудіо файли
        self.samples = []
        speech_commands_dir = os.path.join(root_dir, "SpeechCommands", "speech_commands_v0.02")
        
        if not os.path.exists(speech_commands_dir):
            print(f"❌ Папка {speech_commands_dir} не існує!")
            return
            
        for class_name in self.classes:
            class_dir = os.path.join(speech_commands_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith('.wav'):
                        filepath = os.path.join(class_dir, filename)
                        self.samples.append((filepath, class_name))
        
        print(f"📦 Завантажено {len(self.samples)} зразків для класів {self.classes}")
        
        # Показуємо розподіл
        class_counts = Counter([sample[1] for sample in self.samples])
        print("Розподіл класів:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, class_name = self.samples[idx]
        
        # Завантажуємо аудіо файл
        waveform, sample_rate = sf.read(filepath)
        
        # Перетворюємо в tensor
        if len(waveform.shape) == 1:
            waveform = waveform[np.newaxis, :]  # Додаємо channel dimension
        waveform = torch.from_numpy(waveform).float()
        
        # Застосовуємо обробку
        if self.transform:
            waveform = self.transform(waveform)
        
        # Повертаємо індекс класу
        label_idx = self.class_to_idx[class_name]
        
        return waveform, label_idx

class SimpleAudioPreprocessor:
    """Простий обробник аудіо"""
    
    def __init__(self, sample_rate=16000, n_mels=64):
        self.sample_rate = sample_rate
        self.target_length = sample_rate  # 1 секунда
        
        # Створюємо мел-спектрограму transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
        
        print(f"🔧 Створено обробник: {sample_rate}Hz, {n_mels} мел-фільтрів")
    
    def __call__(self, waveform):
        # Нормалізуємо довжину
        current_length = waveform.shape[-1]
        
        if current_length > self.target_length:
            # Обрізаємо
            waveform = waveform[..., :self.target_length]
        elif current_length < self.target_length:
            # Доповнюємо нулями
            pad_length = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        # Створюємо спектрограму
        mel_spec = self.mel_spectrogram(waveform)
        log_mel = self.amplitude_to_db(mel_spec)
        
        # Простий min-max scaling
        spec_min = log_mel.min()
        spec_max = log_mel.max()
        if spec_max > spec_min:
            normalized = (log_mel - spec_min) / (spec_max - spec_min)
        else:
            normalized = log_mel
            
        return normalized

def create_simple_data_loaders(root_dir="./data", batch_size=32):
    """Створює прості data loaders"""
    
    print("🔧 Створюємо прості data loaders...")
    
    # Обробник
    preprocessor = SimpleAudioPreprocessor()
    
    # Dataset
    dataset = SimpleSpeechCommandsDataset(root_dir, transform=preprocessor)
    
    if len(dataset) == 0:
        print("❌ Не знайдено жодного файлу!")
        return None, None
    
    # Розділяємо на train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Створено loaders:")
    print(f"   🚂 Train: {len(train_loader)} batches ({train_size} зразків)")
    print(f"   🧪 Test: {len(test_loader)} batches ({test_size} зразків)")
    
    return train_loader, test_loader

if __name__ == "__main__":
    print("🧪 Тестуємо простий data loader...")
    
    train_loader, test_loader = create_simple_data_loaders()
    
    if train_loader is not None:
        # Тестуємо один батч
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"\n📊 Тестовий батч:")
            print(f"   Розмір даних: {data.shape}")
            print(f"   Мітки: {target}")
            print(f"   Діапазон даних: {data.min():.3f} - {data.max():.3f}")
            break
        
        print("\n✅ Простий data loader працює!")
        print("🎯 Можемо переходити до створення моделі!")
    else:
        print("❌ Проблема з data loader'ом")