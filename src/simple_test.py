"""
Простий тест обробки даних без візуалізації
"""

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import os
import sys

# Додаємо шлях до модулів
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import SubsetSC

def simple_test():
    """Простий тест без візуалізації"""
    print("🧪 Простий тест обробки даних...")
    
    # Завантажуємо невеликий зразок
    dataset = SubsetSC("training")
    waveform, sample_rate, label, _, _ = dataset[0]
    
    print(f"📊 Оригінальний сигнал:")
    print(f"   Форма: {waveform.shape}")
    print(f"   Частота: {sample_rate} Hz")
    print(f"   Клас: {label}")
    
    # Створюємо мел-спектрограму
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=512
    )
    
    # Нормалізуємо довжину
    target_length = 16000  # 1 секунда
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        pad_length = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    
    # Перетворюємо в спектрограму
    mel_spec = mel_transform(waveform)
    
    print(f"📈 Мел-спектрограма:")
    print(f"   Форма: {mel_spec.shape}")
    print(f"   Діапазон: {mel_spec.min():.3f} - {mel_spec.max():.3f}")
    
    # Логарифмічне масштабування
    amplitude_to_db = T.AmplitudeToDB()
    log_mel = amplitude_to_db(mel_spec)
    
    print(f"📊 Логарифмічна спектрограма:")
    print(f"   Форма: {log_mel.shape}")
    print(f"   Діапазон: {log_mel.min():.3f} - {log_mel.max():.3f}")
    
    return True

if __name__ == "__main__":
    try:
        success = simple_test()
        if success:
            print("\n✅ Тест пройдено успішно!")
            print("🎯 Готові до створення нейронної мережі!")
        
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()