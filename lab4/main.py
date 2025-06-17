from filecmp import cmp
import math
import numpy as np
from matplotlib import pyplot as plt


def bisection_method(f, a, b, tol=1e-3, max_iter=100):
    n = 0
    if f(a) * f(b) >= 0:
        raise ValueError("Функция должна иметь разные знаки на концах отрезка " , a ,b)

    for _ in range(max_iter):
        n+=1
        c = (a + b) / 2
        if b-a < 2*tol:
            return c,n
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2,n
def newton_method(f, f_prime, f_prime2, a, b, tol=1e-3, max_iter=100):
    # Выбираем начальное приближение по условию f(x)*f''(x) > 0
    if f(a) * f_prime2(a) > 0:
        x0 = a
    elif f(b) * f_prime2(b) > 0:
        x0 = b
    else:
        raise ValueError("Не удалось найти подходящее начальное приближение на границах интервала",f(a) ,f_prime2(a), f(b),f_prime2(b))
    x_prev = x0 
    x = x0
    n = 0
    
    for _ in range(max_iter):
        n += 1
        fx = f(x)
        fpx = f_prime(x)
        # print(fx)
        
        if fx * f(x + np.sign(x-x_prev)*tol) < 0:
            return x, n
            
        if fpx == 0:
            raise ValueError("Производная равна нулю. Метод Ньютона не может быть применен.")
        x_prev = x 
        x = x - fx / fpx
        
        # Проверка, что x остался в пределах интервала
        if x < a or x > b:
            raise ValueError("Итерация вышла за пределы интервала")
    
    return x, n
# Найдем корень уравнения x^2 - 2 = 0 (это √2 ≈ 1.41421356)
def f(x):
    return x**3-2*x**2-4*x+3
def f_prime(x):
    return 3*x**2-4*x-4
def f_pp(x):
    return 5*x-4
interval= [[-10,-1],[0,0.65],[2.1,10]]
true_roots = [-1/2 - math.sqrt(5)/2,math.sqrt(5)/2 - 1/2 ,3]

print("x & x_N & error_N & x_B & error_B " )

for ab,res in zip(interval,true_roots):
    a= ab[0]
    b= ab[1]
    root_bisect, n_bisect = bisection_method(f, a, b)
    root_newton, n_newton = newton_method(f, f_prime, f_pp , a, b)

    print(f"{res:10.6f} & {root_bisect:10.6f} & {abs(root_bisect - res):10.6f} & {root_newton:10.6f} & {abs(root_newton - res):10.6f} \\\\")

a= 2
b= 100
# Диапазон tolerances для тестирования
tolerances = [10**(-i) for i in range(1, 12)]

# Собираем данные для графиков
bisection_iterations = []
newton_iterations = []
bisection_errors = []
newton_errors = []
true_root = 3
for tol in tolerances:
    # Метод бисекции
    root_bisect, n_bisect = bisection_method(f, a, b, tol)
    bisection_iterations.append(n_bisect)
    bisection_errors.append(abs(root_bisect - true_root))
    
    # Метод Ньютона
    root_newton, n_newton = newton_method(f, f_prime, f_pp , a, b, tol)
    newton_iterations.append(n_newton)
    newton_errors.append(abs(root_newton - true_root))
print("\nСравнение методов:")
print("Tolerance\tBisect Iter\tNewton Iter\tBisect Error\tNewton Error")
for i in range(len(tolerances)):
    print(f"{tolerances[i]:.1e}\t\t{bisection_iterations[i]}\t\t{newton_iterations[i]}\t\t{bisection_errors[i]:.2e}\t\t{newton_errors[i]:.2e}")


# Построение графиков
plt.figure(figsize=(12, 10))
# График зависимости числа итераций от tolerance
plt.subplot(2, 2, 1)
plt.loglog(tolerances, bisection_iterations, 'b-o', label='Bisection')
plt.loglog(tolerances, newton_iterations, 'r-o', label='Newton')
plt.xlabel('Tolerance')
plt.ylabel('Number of iterations')
plt.title('Iterations vs Tolerance')
plt.legend()
plt.grid(True)

# График зависимости ошибки от tolerance
plt.subplot(2, 2, 2)
plt.loglog(tolerances, bisection_errors, 'b-o', label='Bisection Error')
plt.loglog(tolerances, newton_errors, 'r-o', label='Newton Error')
plt.loglog(tolerances, tolerances, 'g--', label='Tolerance line')
plt.xlabel('Tolerance')
plt.ylabel('Absolute Error')
plt.title('Error vs Tolerance')
plt.legend()
plt.grid(True)

# График зависимости ошибки от числа итераций
plt.subplot(2, 2, 3)
plt.semilogy(bisection_iterations, bisection_errors, 'b-o', label='Bisection')
plt.semilogy(newton_iterations, newton_errors, 'r-o', label='Newton')
plt.xlabel('Number of iterations')
plt.ylabel('Absolute Error')
plt.title('Error vs Iterations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Вывод таблицы сравнения

