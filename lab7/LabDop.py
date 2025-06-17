import cmath
import math
import matplotlib.pyplot as plt
import numpy as np
pi=math.pi
# Определяем функцию f(x) = exp(sin(2πx))
def f(x):
    # return abs(x-0.5)
    # return abs(math.sin(2*x*pi))
    # return math.sin(2*x*pi)
    # return math.exp(math.sin(2 * math.pi * x))
    return x
    # return np.sign(math.sin(2*x*pi))

def fft_manual(x):
    n = len(x)
    if n <= 1:
        return x

    if n & (n - 1) != 0:
        raise ValueError("Длина входного массива должна быть степенью двойки")

    if n == 2:
        return [x[0] + x[1], x[0] - x[1]]

    even = fft_manual(x[0::2])
    odd = fft_manual(x[1::2])

    y = [0] * n
    for k in range(n // 2):
        t = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        y[k] = even[k] + t
        y[k + n // 2] = even[k] - t

    return y
L=0
L=2**L
# Параметры
N = 128  # Количество узлов для БПФ
x_j = [j*L / N for j in range(N)]  # Узлы x_j = j/N
f_x = [f(x) for x in x_j]  # Значения f(x) в узлах

# Вычисляем коэффициенты Фурье с помощью БПФ
c_k = fft_manual(f_x)
c_k = [x / N for x in c_k]  # Нормализация

y_j = [0.5 + j*L / N for j in range(N)]
f_true = [f(y) for y in y_j]  # Истинные значения f(y)
f_interp = [0] * N
for j in range(N):
    for k in range(N):
        f_interp[j] += c_k[k] * cmath.exp(2j * cmath.pi * k * y_j[j]/L)
    f_interp[j] = f_interp[j].real  # Берем действительную часть

# Сравнение
print("Сравнение в средних точках y_j (первые 5 значений):")
print("j\tИстинное f(y)\tИнтерполяция\tОшибка")
for j in range(5):
    error = abs(f_true[j] - f_interp[j])
    print(f"{j}\t{f_true[j]:.6f}\t\t{f_interp[j]:.6f}\t\t{error:.6e}")
# Плотная сетка для вывода

M = N  # Количество точек для графика
y_j = [j*L / M for j in range(M)]  # Сетка y_j

# Истинная функция
f_true = [f(y) for y in y_j]

# Вычисляем вклад каждой гармоники и их сумму
num_harmonics = 17  # Ограничимся первыми 5 гармониками для наглядности
harmonics = [[0] * M for _ in range(num_harmonics)]  # Вклад каждой гармоники
f_interp = [0] * M  # Сумма всех гармоник (полная интерполяция)

# Для каждой точки y_j вычисляем вклад каждой гармоники
for j in range(M):
    for k in range(N):
        # Вклад k-й гармоники: c_k * e^(2πi k y_j)
        term = c_k[k] * cmath.exp(2j * cmath.pi * k * y_j[j]/L)
        # Суммируем для полной интерполяции
        f_interp[j] += term.real
        # Сохраняем вклад первых num_harmonics гармоник
        if k < num_harmonics:
            harmonics[k][j] = term.real

# Построение графика
plt.figure(figsize=(12, 7))
# Истинная функция
plt.plot(y_j, f_true, 'b-', label='Истинная функция ', linewidth=2)
# Вклад каждой гармоники
colors = [
    'green', 'cyan', 'magenta', 'yellow', 'black',
    'orange', 'purple', 'lime', 'teal', 'navy', 'maroon',
    'olive', 'silver', 'gold', 'pink', 'brown', 'gray'
]  # Разные цвета для гармоник
for k in range(num_harmonics):
    plt.plot(y_j, harmonics[k], linestyle='--', color=colors[k], 
             label=f'Гармоника k={k}', alpha=0.7)
# Полная интерполяция (сумма всех гармоник)
plt.plot(y_j, f_interp, 'r--', label='Сумма всех гармоник (БПФ)', linewidth=2)

# Настройка графика
plt.xlabel('$y$', fontsize=12)
plt.ylabel('Значения', fontsize=12)
plt.title('Вклад отдельных гармоник и их сумма (Тригонометрическая интерполяция с БПФ)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate('Интерполяция выполнена с использованием\nБыстрого преобразования Фурье (БПФ)', 
             xy=(0.5, 0.95), xycoords='axes fraction', 
             ha='center', va='top', fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
plt.tight_layout()
plt.show()

# Вывод максимальной ошибки
max_error = max(abs(f_true[j] - f_interp[j]) for j in range(M))
print(f"Максимальная ошибка: {max_error:.6e}")