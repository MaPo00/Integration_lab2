#!/usr/bin/env python3
"""
🔬 Benchmark порівняння Docker vs Local версій Speech Commands API
Виконує серію тестів для порівняння продуктивності
"""

import time
import requests
import statistics
import json
from typing import Dict, List
import sys
import subprocess
import psutil
import os

class APIBenchmark:
    def __init__(self):
        self.docker_url = "http://localhost:8000"
        self.local_url = "http://localhost:5000"
        self.test_payload = {"text": "benchmark test"}
        
    def check_api_availability(self, url: str, name: str) -> bool:
        """Перевіряє доступність API"""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} API доступний на {url}")
                return True
            else:
                print(f"❌ {name} API недоступний (статус: {response.status_code})")
                return False
        except Exception as e:
            print(f"❌ {name} API недоступний: {e}")
            return False
    
    def measure_latency(self, url: str, num_requests: int = 10) -> List[float]:
        """Вимірює латентність для серії запитів"""
        latencies = []
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{url}/predict_text", 
                    json=self.test_payload,
                    timeout=10
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                else:
                    print(f"⚠️ Запит {i+1} повернув статус {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Помилка в запиті {i+1}: {e}")
        
        return latencies
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Отримує використання пам'яті"""
        # Для Docker контейнера
        try:
            result = subprocess.run(
                ["docker", "stats", "speech-api-v2", "--no-stream", "--format", "table {{.MemUsage}}"],
                capture_output=True, text=True, timeout=10
            )
            docker_mem = result.stdout.strip().split('\n')[-1] if result.returncode == 0 else "N/A"
        except:
            docker_mem = "N/A"
        
        # Для локального процесу
        current_process = psutil.Process()
        local_mem_mb = current_process.memory_info().rss / 1024 / 1024
        
        return {
            "docker_memory": docker_mem,
            "local_memory_mb": round(local_mem_mb, 2)
        }
    
    def get_disk_usage(self) -> Dict[str, str]:
        """Отримує використання дискового простору"""
        # Розмір Docker образу
        try:
            result = subprocess.run(
                ["docker", "images", "speech-commands-api:v2", "--format", "table {{.Size}}"],
                capture_output=True, text=True, timeout=10
            )
            docker_size = result.stdout.strip().split('\n')[-1] if result.returncode == 0 else "N/A"
        except:
            docker_size = "N/A"
        
        # Розмір локального проекту
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(".."):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
        
        local_size_mb = round(total_size / 1024 / 1024, 2)
        
        return {
            "docker_image_size": docker_size,
            "local_project_size_mb": local_size_mb
        }
    
    def run_benchmark(self) -> Dict:
        """Запускає повний benchmark"""
        print("🔬 Запускаємо benchmark порівняння Docker vs Local")
        print("=" * 60)
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "docker": {"available": False},
            "local": {"available": False}
        }
        
        # Перевіряємо доступність API
        docker_available = self.check_api_availability(self.docker_url, "Docker")
        local_available = self.check_api_availability(self.local_url, "Local")
        
        results["docker"]["available"] = docker_available
        results["local"]["available"] = local_available
        
        if not (docker_available or local_available):
            print("❌ Жоден API не доступний для тестування!")
            return results
        
        print("\n📊 Тестуємо латентність (10 запитів до кожного API)...")
        
        # Тестуємо Docker API
        if docker_available:
            print("\n🐳 Тестуємо Docker API...")
            docker_latencies = self.measure_latency(self.docker_url)
            if docker_latencies:
                results["docker"].update({
                    "latency_ms": {
                        "mean": round(statistics.mean(docker_latencies), 2),
                        "median": round(statistics.median(docker_latencies), 2),
                        "min": round(min(docker_latencies), 2),
                        "max": round(max(docker_latencies), 2),
                        "std": round(statistics.stdev(docker_latencies) if len(docker_latencies) > 1 else 0, 2)
                    },
                    "success_rate": len(docker_latencies) / 10
                })
        
        # Тестуємо Local API
        if local_available:
            print("\n💻 Тестуємо Local API...")
            local_latencies = self.measure_latency(self.local_url)
            if local_latencies:
                results["local"].update({
                    "latency_ms": {
                        "mean": round(statistics.mean(local_latencies), 2),
                        "median": round(statistics.median(local_latencies), 2),
                        "min": round(min(local_latencies), 2),
                        "max": round(max(local_latencies), 2),
                        "std": round(statistics.stdev(local_latencies) if len(local_latencies) > 1 else 0, 2)
                    },
                    "success_rate": len(local_latencies) / 10
                })
        
        print("\n💾 Збираємо інформацію про ресурси...")
        
        # Збираємо інформацію про ресурси
        memory_info = self.get_memory_usage()
        disk_info = self.get_disk_usage()
        
        results["resources"] = {
            "memory": memory_info,
            "disk": disk_info
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Виводить результати у читабельному форматі"""
        print("\n" + "=" * 80)
        print("📋 РЕЗУЛЬТАТИ BENCHMARK ТЕСТУВАННЯ")
        print("=" * 80)
        
        if results["docker"]["available"]:
            print("\n🐳 DOCKER КОНТЕЙНЕР:")
            if "latency_ms" in results["docker"]:
                lat = results["docker"]["latency_ms"]
                print(f"   ⚡ Середня латентність: {lat['mean']} мс")
                print(f"   📊 Медіана: {lat['median']} мс")
                print(f"   ⬇️ Мінімум: {lat['min']} мс")
                print(f"   ⬆️ Максимум: {lat['max']} мс")
                print(f"   📈 Стандартне відхилення: {lat['std']} мс")
                print(f"   ✅ Успішність: {results['docker']['success_rate']*100}%")
        
        if results["local"]["available"]:
            print("\n💻 ЛОКАЛЬНА ВЕРСІЯ:")
            if "latency_ms" in results["local"]:
                lat = results["local"]["latency_ms"]
                print(f"   ⚡ Середня латентність: {lat['mean']} мс")
                print(f"   📊 Медіана: {lat['median']} мс")
                print(f"   ⬇️ Мінімум: {lat['min']} мс")
                print(f"   ⬆️ Максимум: {lat['max']} мс")
                print(f"   📈 Стандартне відхилення: {lat['std']} мс")
                print(f"   ✅ Успішність: {results['local']['success_rate']*100}%")
        
        print("\n💾 ВИКОРИСТАННЯ РЕСУРСІВ:")
        if "resources" in results:
            res = results["resources"]
            print(f"   🐳 Docker образ: {res['disk']['docker_image_size']}")
            print(f"   💻 Локальний проект: {res['disk']['local_project_size_mb']} MB")
            print(f"   🧠 Docker пам'ять: {res['memory']['docker_memory']}")
            print(f"   🧠 Локальна пам'ять: {res['memory']['local_memory_mb']} MB")
        
        # Порівняння швидкості
        if (results["docker"]["available"] and results["local"]["available"] and
            "latency_ms" in results["docker"] and "latency_ms" in results["local"]):
            
            docker_mean = results["docker"]["latency_ms"]["mean"]
            local_mean = results["local"]["latency_ms"]["mean"]
            
            print("\n🏆 ПОРІВНЯННЯ:")
            if docker_mean < local_mean:
                speedup = round((local_mean / docker_mean - 1) * 100, 1)
                print(f"   🐳 Docker швидший на {speedup}%")
            else:
                slowdown = round((docker_mean / local_mean - 1) * 100, 1)
                print(f"   💻 Local швидший на {slowdown}%")
        
        print("=" * 80)
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Зберігає результати у файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Результати збережено у {filename}")

def main():
    benchmark = APIBenchmark()
    results = benchmark.run_benchmark()
    benchmark.print_results(results)
    benchmark.save_results(results)

if __name__ == "__main__":
    main()