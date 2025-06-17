
""" y''+py'+qy = f(x)
    y(0)=y0
    y'(0)=y0'

    y1 = y
    y2 = y'
    
    y1' = y' = y2
    y1' = y2

    y2' =y'' =f(x)-p(x)y2-q(x)y1

    =>
    y1(0)=y0
    y2(0)=y0'

    y1'=f1(x,y1,y2) = y2
    y2'=f2(x,y1,y2) = -q(x)y1-p(x)y2+f(x)
"""
import math
import matplotlib.pyplot as plt
import random
import numpy as np


def runge_kutta_step(p, q, f, x, y1, y2, h):
    """Один шаг метода Рунге-Кутты 4-го порядка"""
    def f1(x, y1, y2):
        return y2
    
    def f2(x, y1, y2):
        return f(x) - p(x) * y2 - q(x) * y1
    
    k1_1 = h * f1(x, y1, y2)
    k2_1 = h * f2(x, y1, y2)
    
    k1_2 = h * f1(x + h/2, y1 + k1_1/2, y2 + k2_1/2)
    k2_2 = h * f2(x + h/2, y1 + k1_1/2, y2 + k2_1/2)
    
    k1_3 = h * f1(x + h/2, y1 + k1_2/2, y2 + k2_2/2)
    k2_3 = h * f2(x + h/2, y1 + k1_2/2, y2 + k2_2/2)
    
    k1_4 = h * f1(x + h, y1 + k1_3, y2 + k2_3)
    k2_4 = h * f2(x + h, y1 + k1_3, y2 + k2_3)
    
    y1_new = y1 + (k1_1 + 2*k1_2 + 2*k1_3 + k1_4) / 6
    y2_new = y2 + (k2_1 + 2*k2_2 + 2*k2_3 + k2_4) / 6
    
    return y1_new, y2_new
def runge_kutta_4th_order_system(p, q, f, y0, y0_prime, x_range, h,isNeedR = False):
    """
    Решает ОДУ второго порядка y'' + p(x)y' + q(x)y = f(x) методом Рунге-Кутты 4-го порядка.
    
    Параметры:
        p (function): Функция p(x).
        q (function): Функция q(x).
        f (function): Функция f(x).
        y0 (float): Начальное условие y(0).
        y0_prime (float): Начальное условие y'(0).
        x_range (tuple): Интервал (x, x_end).
        h (float): Шаг интегрирования.
        
    Возвращает:
        x_values (ndarray): Массив значений x.
        y_values (ndarray): Массив значений y.
    """
    p_order = 4
    # Инициализация массивов
    x_values = []
    y1_values = [] # y1 = y
    y2_values = []  # y2 = y'
    # Начальные условия
    x_values.append(x_range[0])
    y1_values.append(y0)
    y2_values.append(y0_prime)
    
    # Функции системы ОДУ первого порядка
    def f1(x, y1, y2):
        return y2
    
    def f2(x, y1, y2):
        return f(x) - p(x) * y2 - q(x) * y1
    
    # Метод Рунге-Кутты 4-го порядка
    while  x_values[-1] < x_range[1]:
        x = x_values[-1]
        y1 = y1_values[-1]
        y2 = y2_values[-1]
        
        # Обновление значений
        # Делаем два шага h
        y1_h, y2_h = runge_kutta_step(p, q, f, x, y1, y2, h)
        y1_2h, y2_2h = runge_kutta_step(p, q, f, x + h, y1_h, y2_h, h)
        
        # Делаем один шаг 2h
        y1_2h_single, y2_2h_single = runge_kutta_step(p, q, f, x, y1, y2, 2*h)
        
        # Оценка погрешности
        # err = (y1_2h - y1_2h_single)**2+(y2_2h - y2_2h_single)**2
        err = max(abs(y1_2h - y1_2h_single),abs(y2_2h - y2_2h_single))
        err_r = err / (2**p_order - 1)  
        h_opt = h * (eps / (err_r + 1e-15))**(1/(p_order + 1))
        # print(err_r,h,x)
        if (err_r  <= eps): 
            # x_values.append(x + h)
            x_values.append(x + 2*h)
            if (not isNeedR) :
                # y1_values.append(y1_h)
                y1_values.append(y1_2h)
                # y2_values.append(y2_h)
                y2_values.append(y2_2h)
            else:
                # y1_values.append(y1_h+ ( y1_h  - y1_2h_single)/(2**p_order-1))
                y1_values.append(y1_2h+ ( y1_2h  - y1_2h_single)/(2**p_order-1))
                # y2_values.append(y2_h + ( y2_2h  - y2_2h_single)/(2**p_order-1))
                y2_values.append(y2_2h + ( y2_2h  - y2_2h_single)/(2**p_order-1))
            
        # h = h/(2.0)
        h = 0.9 * h_opt  # Берем немного меньший шаг для надежности

    return x_values, y1_values
def runge_kutta_4th_order_system2(p, q, f, y0, y0_prime, x_range, h,isNeedR = False):
    """
    Решает ОДУ второго порядка y'' + p(x)y' + q(x)y = f(x) методом Рунге-Кутты 4-го порядка.
    
    Параметры:
        p (function): Функция p(x).
        q (function): Функция q(x).
        f (function): Функция f(x).
        y0 (float): Начальное условие y(0).
        y0_prime (float): Начальное условие y'(0).
        x_range (tuple): Интервал (x, x_end).
        h (float): Шаг интегрирования.
        
    Возвращает:
        x_values (ndarray): Массив значений x.
        y_values (ndarray): Массив значений y.
    """
    p_order = 4
    # Инициализация массивов
    x_values = []
    y1_values = [] # y1 = y
    y2_values = []  # y2 = y'
    # Начальные условия
    x_values.append(x_range[0])
    y1_values.append(y0)
    y2_values.append(y0_prime)
    
    # Функции системы ОДУ первого порядка
    def f1(x, y1, y2):
        return y2
    
    def f2(x, y1, y2):
        return f(x) - p(x) * y2 - q(x) * y1
    
    # Метод Рунге-Кутты 4-го порядка
    while  x_values[-1] < x_range[1]:
        x = x_values[-1]
        y1 = y1_values[-1]
        y2 = y2_values[-1]
        
        # Обновление значений
        # Делаем два шага h
        y1_h, y2_h = runge_kutta_step(p, q, f, x, y1, y2, h)
        y1_2h, y2_2h = runge_kutta_step(p, q, f, x + h, y1_h, y2_h, h)
        
        # Делаем один шаг 2h
        y1_2h_single, y2_2h_single = runge_kutta_step(p, q, f, x, y1, y2, 2*h)
        
        # Оценка погрешности
        # err = (y1_2h - y1_2h_single)**2+(y2_2h - y2_2h_single)**2
        err = max(abs(y1_2h - y1_2h_single),abs(y2_2h - y2_2h_single))
        err_r = err / (2**p_order - 1)  
        # h_opt = h * (eps / (err_r + 1e-15))**(1/(p_order + 1))
        # print(err_r,h,x)
        if (err_r  <= eps): 
            # x_values.append(x + h)
            x_values.append(x + 2*h)
            if (not isNeedR) :
                # y1_values.append(y1_h)
                y1_values.append(y1_2h)
                # y2_values.append(y2_h)
                y2_values.append(y2_2h)
            else:
                # y1_values.append(y1_h+ ( y1_h  - y1_2h_single)/(2**p_order-1))
                y1_values.append(y1_2h+ ( y1_2h  - y1_2h_single)/(2**p_order-1))
                # y2_values.append(y2_h + ( y2_2h  - y2_2h_single)/(2**p_order-1))
                y2_values.append(y2_2h + ( y2_2h  - y2_2h_single)/(2**p_order-1))
            
        h = h/(2.0)
        # h = 0.9 * h_opt  # Берем немного меньший шаг для надежности

    return x_values, y1_values
def runge_kutta_4th_order_system3(p, q, f, y0, y0_prime, x_range, h,isNeedR = False):
    """
    Решает ОДУ второго порядка y'' + p(x)y' + q(x)y = f(x) методом Рунге-Кутты 4-го порядка.
    
    Параметры:
        p (function): Функция p(x).
        q (function): Функция q(x).
        f (function): Функция f(x).
        y0 (float): Начальное условие y(0).
        y0_prime (float): Начальное условие y'(0).
        x_range (tuple): Интервал (x, x_end).
        h (float): Шаг интегрирования.
        
    Возвращает:
        x_values (ndarray): Массив значений x.
        y_values (ndarray): Массив значений y.
    """
    h=h/2
    error = 0
    p_order = 4
    # Инициализация массивов
    x_values = []
    y1_values = [] # y1 = y
    y2_values = []  # y2 = y'
    # Начальные условия
    x_values.append(x_range[0])
    y1_values.append(y0)
    y2_values.append(y0_prime)
    
    # Функции системы ОДУ первого порядка
    def f1(x, y1, y2):
        return y2
    
    def f2(x, y1, y2):
        return f(x) - p(x) * y2 - q(x) * y1
    
    # Метод Рунге-Кутты 4-го порядка
    while  x_values[-1] < x_range[1]:
        x = x_values[-1]
        y1 = y1_values[-1]
        y2 = y2_values[-1]
        y1_h, y2_h = runge_kutta_step(p, q, f, x, y1, y2, h)
        y1_2h, y2_2h = runge_kutta_step(p, q, f, x + h, y1_h, y2_h, h)
        y1_2h_single, y2_2h_single = runge_kutta_step(p, q, f, x, y1, y2, 2*h) 
        # err = (y1_2h - y1_2h_single)**2+(y2_2h - y2_2h_single)**2
        err = max(abs(y1_2h - y1_2h_single),abs(y2_2h - y2_2h_single))
        err_r = abs(err)/(15)
        error = max(error,err_r)
        x_values.append(x + 2*h)
        if (not isNeedR) :
            y1_values.append(y1_2h)
            y2_values.append(y2_2h)
        else:
            y1_values.append(y1_2h+ ( y1_2h  - y1_2h_single)/(2**p_order-1))
            y2_values.append(y2_2h + ( y2_2h  - y2_2h_single)/(2**p_order-1))
    return x_values, y1_values , error
def runge_cicle(p, q, f, y0, y0_prime, x_range, h,isNeedR = False):
    errors = []
    hs= []
    while True: 
        x_value,y_value,error1  = runge_kutta_4th_order_system3(p, q, f, y0, y0_prime, x_range, h,isNeedR)
        errors.append(error1)
        hs.append(1/h)
        if ( error1 < eps):
            return x_value,y_value,errors,hs
        h=h/2
     
if __name__ == "__main__":
    # Пример: y'' - y = e^x (решение y = (2 + 0.5x)e^x + e^(-x))
    p = lambda x: -4
    q = lambda x: 3
    f = lambda x: math.exp(5*x)
    y0 = 3
    y0_prime = 9
    
    x_range = (0, 1)
    eps = 10**(-3)
    h = 1
    
    # Решение без уточнения
    x_values, y_values,cl_errors,hs = runge_cicle(p, q, f, y0, y0_prime, x_range, h)
    # Решение с уточнением Ричардсона
    x_values_r, y_values_r,cl_errors,hs = runge_cicle(p, q, f, y0, y0_prime, x_range, h, True)
    def y(x):
        return (1/8) * math.exp(x) * (22 * math.exp(2*x) + math.exp(4*x) + 1)
    # Точное решение для сравнения
    # exact_solution = [(2 + 0.5*x)*math.exp(x) + math.exp(-x) for x in x_values]
    exact_solution = [y(x) for x in x_values]
    # Вычисление ошибок
    errors = [abs(y - exact) for y, exact in zip(y_values, exact_solution)]
    
    errors_r = [abs(y - exact) for y, exact in zip(y_values_r, exact_solution)]
    print("x_values[i],exact_solution[i],y_values[i],errors[i],y_valuesR[i],errorsR[i]")
    for i, _ in enumerate(x_values):
        print(f"{x_values[i]:.5f} & {exact_solution[i]:.5f} & {y_values[i]:.5f} & {errors[i]:.5f}  & {errors_r[i]:.5f} \\\\")
    print("max_error= ",max(errors))
    print("max_errorR= ",max(errors_r))
    print("n= ",len(errors))
    print("h= ",(x_values[1]-x_values[0]))
    print("h**5= ",(x_values[1]-x_values[0])**5)
    # Визуализация решений и ошибок
    plt.figure(figsize=(12, 8))
    
    # График решений
    plt.subplot(3, 1, 1)
    plt.plot(x_values, y_values, 'bo', label="Численное решение")
    plt.plot(x_values_r, y_values_r, 'go', label="Численное решение c уточнением ")
    plt.plot(x_values, exact_solution, 'r:', label="Точное решение")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Сравнение решений")
    plt.legend()
    plt.grid()
    
    # График ошибок
    plt.subplot(3, 1, 2)
    plt.plot(x_values, errors, 'ro', label="Абсолютная ошибка на последнем цикле Рунге Кнута ")
    plt.xlabel("x")
    plt.ylabel("Абсолютная ошибка")
    plt.title("График абсолютной ошибок")
    plt.yscale('log')  # Логарифмическая шкала для ошибок
    plt.legend()
    plt.grid()
    
    # График ошибок
    plt.subplot(3, 1, 3)
    plt.plot(hs, cl_errors, 'ro', label="Ошибка по Рунге ")
    plt.xlabel("n")
    plt.ylabel("Ошибка по Рунге ")
    plt.title("График ошибок")
    plt.yscale('log')  # Логарифмическая шкала для ошибок
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()