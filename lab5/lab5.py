import numpy as np
import matplotlib.pyplot as plt

# Target function
def f(x):
    x1, x2 = x[0], x[1]
    return 2 * x1**2 + x1 * x2 + 3 * x2**2 + x1 + 2 * x2

# Gradient of the function
def grad_f(x):
    x1, x2 = x[0], x[1]
    df_dx1 = 4 * x1 + x2 + 1
    df_dx2 = x1 + 6 * x2 + 2
    return np.array([df_dx1, df_dx2])

# Golden section search for line search
def golden_section_search(phi, a=0, b=1, tol=1e-3):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    while abs(b - a) > tol:
        if phi(c) < phi(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2

# Steepest descent method with golden section search
def steepest_descent_with_golden(f, grad_f, x0, tol=1e-3, max_iter=100):
    x = x0
    trajectory = [x]
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
        phi = lambda t: f(x - t * grad)
        t = golden_section_search(phi, a=0, b=1, tol=1e-3)
        
        x_new = x - t * grad
        trajectory.append(x_new)
        
        x = x_new

    return x, f(x), trajectory, iterations

# Run the method
x0 = np.array([0.0, 0.0])
minimum, f_minimum, trajectory, iterations = steepest_descent_with_golden(f, grad_f, x0)

# Analytic solution
analytic_x = np.array([-4/23, -7/23])
analytic_f = f(analytic_x)

# Differences
delta_x = minimum - analytic_x
delta_f = f_minimum - analytic_f

# Output results
print("Start point:")
print(f"  x_0 = ({x0[0]:.1f}, {x0[1]:.1f})")
print("\nNumeric solution:")
print(f"  x = ({minimum[0]:.10f}, {minimum[1]:.10f})")
print(f"  f = {f_minimum:.10f}")
print(f"  iterations = {iterations}")
print("\nAnalytic solution:")
print(f"  x = ({analytic_x[0]:.10f}, {analytic_x[1]:.10f})")
print(f"  f = {analytic_f:.10f}")
print("\nDifferences:")
print(f"  delta_x = ({delta_x[0]:.2e}, {delta_x[1]:.2e})")
print(f"  delta_f = {delta_f:.2e}")

# Plotting
trajectory = np.array(trajectory)
margin = 0.5
x_min, x_max = trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin
y_min, y_max = trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin

x_vals = np.linspace(x_min, x_max, 400)
y_vals = np.linspace(y_min, y_max, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 2 * X**2 + X * Y + 3 * Y**2 + X + 2 * Y

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label='Траектория')
plt.scatter(minimum[0], minimum[1], color='blue', label='Минимум')
plt.scatter(analytic_x[0], analytic_x[1], color='green', marker='x', label='Аналитический минимум')
plt.title('Наискорейший спуск с методом золотого сечения')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.legend()
plt.grid(True)
plt.show()