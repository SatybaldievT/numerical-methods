import math
import random

def rectangle_rule(f, a, b, n):
    h = (b - a) / n
    integral = 0
    for i in range(n):
        integral += f(a + h/2 + i * h)
    integral *= h
    return integral

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        integral += f(a + i * h)
    integral *= h
    return integral

def simpsons_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Количество отрезков n должно быть чётным.")
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1,n):
        x = a + i * h
        if i % 2 == 1:
            integral += 4 * f(x)
        else:  
            integral += 2 * f(x)
    integral *= h / 3
    return integral

def monte_carlo(f, a, b, n):
    """Реализация метода Монте-Карло для численного интегрирования"""
    total = 0
    sum_sq = 0
    
    for _ in range(n):
        x = random.uniform(a, b)
        fx = f(x)
        total += fx
        sum_sq += fx ** 2
    
    integral = (b - a) * total / n
    variance = (sum_sq/n - (total/n)**2) * (b - a)**2 / n
    
    return integral

def integral_find(rule, f, a, b, p):
    n = 4
    result  = rule(f, a, b, n)
    result1 =rule(f, a, b, n*2)
     
    error = (result1 - result)/(2**p-1)
    
    e = 0.001
    while abs(error) >= e and n <=33554432 :
        n *= 2
        result = rule(f, a, b, n)
        result1 =rule(f, a, b, n*2)
        error = (result1 - result)/(2**p-1)
        
    return ({(result1-integral(a,b))/integral(a,b)}, error, result1, result1 + error, integral(a,b), n)

def f(x):
    # return (math.e**(x-2))*(x**2)
    return math.e**(x)

def integral(a, b):
    e = math.e
    result = (2*a -2 - (a**2))*(e**(a-2)) + (2 - 2*b + (b**2))*(e**(b-2))
    return result

a = 0
b = 1


I3 = integral_find(simpsons_rule, f, a, b, 4)
I2 = integral_find(trapezoidal_rule, f, a, b, 2)
I1 = integral_find(rectangle_rule, f, a, b, 2)
I4 = integral_find(monte_carlo, f, a, b, 2) 
print("e = 0.001")
print("I = " ,integral(a, b) )
headers = ["Параметр", "Метод прямоугольников", "Метод Трапеций", "Метод Симпсона", "Метод Монте-Карло"]
data = [
    ["Ошибка отн.мат. интеграла", I1[0], I2[0], I3[0], I4[0]],
    ["R", I1[1], I2[1], I3[1], I4[1]],
    ["I*", I1[2], I2[2], I3[2], I4[2]],
    ["I* + R", I1[3], I2[3], I3[3], I4[3]],
    ["n", I1[5], I2[5], I3[5], I4[5]]
]

column_widths = [25, 30, 30, 30, 30]
header_row = "|".join(f"{header:^{width}}" for header, width in zip(headers, column_widths))
print(header_row)
print("-" * len(header_row))

for row in data:
    row_str = "|".join(f"{str(item):^{width}}" for item, width in zip(row, column_widths))
    print(row_str)