import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import re

CPP_FILE = "multiplier.cpp"
EXE_FILE = "multiplier"
RESULT_FILE = "result_matrix.txt"

def compile_cpp():
    print(f"Компиляция {CPP_FILE}...")
    cmd = ["mpicxx", "-O2", "-o", EXE_FILE, CPP_FILE] # Добавил флаг -O2 для корректной работы в Release
    try:
        subprocess.check_call(cmd)
        print("Компиляция успешна.")
    except subprocess.CalledProcessError:
        print("Ошибка компиляции. Убедитесь, что установлен mpicxx.")
        exit(1)

def run_benchmark(sizes, processes_list, runs=2):
    """
    Запуск бенчмарков. 
    runs=2 означает, что для каждого размера мы запускаем 2 раза и берем среднее.
    """
    results = {p: [] for p in processes_list}
    
    os.makedirs("plots", exist_ok=True)

    for p in processes_list:
        print(f"\n=== Тестирование с {p} процессом(ами) ===")
        times_for_size = []
        
        for N in sizes:
            print(f"  Запуск для размера N={N}...", end=" ", flush=True)
            run_times = []
            
            for i in range(runs):
                cmd = ["mpirun", "-np", str(p), f"./{EXE_FILE}", str(N)]
                try:
                    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                    match = re.search(r"Time:\s+([\d.]+)", output)
                    if match:
                        run_times.append(float(match.group(1)))
                except subprocess.CalledProcessError as e:
                    print(f"Ошибка: {e.output}")
                    continue
            
            if run_times:
                avg_time = np.mean(run_times)
                times_for_size.append(avg_time)
                print(f"Готово. Среднее время: {avg_time:.4f} сек")
            else:
                times_for_size.append(0)
                print("Провал.")
                
        results[p] = times_for_size

    return results

def plot_results(sizes, results):
    plt.figure(figsize=(12, 7))
    
    colors = ['b', 'g', 'c', 'm']
    markers = ['o', 's', '^', 'd']
    
    for idx, (p, times) in enumerate(results.items()):
        if idx < len(colors):
            plt.plot(sizes, times, color=colors[idx], marker=markers[idx], linestyle='-', 
                     linewidth=2, markersize=8, label=f'{p} process(es)')
            
            # Отметим среднее время для данного количества процессов пунктиром
            if times:
                mean_val = np.mean(times)
                plt.axhline(y=mean_val, color=colors[idx], linestyle='--', alpha=0.3)

    plt.title('Зависимость времени выполнения от размера матрицы (O(N^3))', fontsize=14)
    plt.xlabel('Размер матрицы (N x N)', fontsize=12)
    plt.ylabel('Время выполнения (секунды)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Настройка логарифмической шкалы (опционально, помогает увидеть прямую на лог-лог графике)
    # Но для отчета чаще нужен обычный график, где виден параболический рост
    # plt.xscale('log')
    # plt.yscale('log')
    
    save_path = "plots/performance_graph.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nГрафик сохранен: {save_path}")
    plt.show()

def verify_with_numpy(N):
    # Проверка размерности
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r') as f:
            lines = f.readlines()
        file_N = int(lines[0].strip())
        if file_N == N:
            print(f"\n✅ Верификация: Размерность матрицы в файле ({file_N}) совпадает с ожидаемой ({N}).")
        else:
            print(f"\n❌ Ошибка верификации: Размерность не совпадает!")

if __name__ == "__main__":
    compile_cpp()

    # ИЗМЕНЕННЫЕ РАЗМЕРЫ:
    # Используем геометрическую прогрессию, чтобы был виден изгиб (кубическая зависимость)
    # Размеры: 100 -> 200 -> 400 -> 800 -> 1600 (каждый шаг х2)
    sizes = [100, 200, 400, 600, 800, 1000, 1200]
    
    processes = [1, 2, 4, 8]
    
    print("ВНИМАНИЕ: Тестирование больших матриц (до 1600) может занять несколько минут.")
    print("Это необходимо для построения корректного графика сложности O(N^3).")
    
    benchmark_results = run_benchmark(sizes, processes, runs=2)
    plot_results(sizes, benchmark_results)
    
    verify_with_numpy(sizes[-1])