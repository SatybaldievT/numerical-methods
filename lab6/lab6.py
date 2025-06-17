import numpy as np
import matplotlib.pyplot as plt

def compute_polynomial(n, x, y, x_a, x_g, x_h, y_a, y_g, y_h):
    """
    Compute coefficients, MSE, and deltas for a polynomial of degree n (manually).
    Returns a dictionary with coefficients, MSE, and deltas.
    """
    # Construct the matrix A and vector b
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            A[i, j] = np.sum(x ** (2*n-i - j ))
        #     print((2*n-i - j ))
        # print()
        b[i] = np.sum(x ** (n-i) * y)
        # print((n-i))
    
    # Solve for coefficients [a_n, a_{n-1}, ..., a_1, a_0]
    coeffs = np.linalg.solve(A, b)
    
    # Compute predicted values
    y_pred = np.zeros_like(x)
    for i in range(n + 1):
        y_pred += coeffs[n-i] * (x ** i)  # coeffs in descending degree order
    
    # Compute MSE
    mse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    # Compute deltas
    z_xa = sum(coeffs[n - i] * x_a ** i for i in range(n + 1))
    z_xg = sum(coeffs[n - i] * x_g ** i for i in range(n + 1))
    z_xh = sum(coeffs[n - i] * x_h ** i for i in range(n + 1))
    deltas = {
        'delta1': abs(z_xa - y_a),
        'delta2': abs(z_xg - y_g),
        'delta3': abs(z_xa - y_g),
        'delta4': abs(z_xg - y_a),
        'delta5': abs(z_xh - y_a),
        'delta6': abs(z_xa - y_h),
        'delta7': abs(z_xh - y_h),
        'delta8': abs(z_xh - y_g),
        'delta9': abs(z_xg - y_h)
    }
    
    # Generate smooth curve for plotting
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = np.zeros_like(x_smooth)
    for i in range(n + 1):
        y_smooth += coeffs[n - i] * x_smooth ** i
    
    # Create and save plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x_smooth, y_smooth, 'r-', label=f'Fitted curve: z (degree {n})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Least Squares Fit: Polynomial degree {n}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'z_polynomial_degree_{n}_plot.png')
    plt.close()
    
    # Prepare results
    result = {'mse': mse, 'deltas': deltas}
    for i in range(n + 1):
        result[f'a_{n-i}'] = coeffs[i]  # Label coefficients as a_n, a_{n-1}, ..., a_0
    
    return result

def compute_least_squares_coefficients(x, y, poly_degree=4):
    """
    Compute coefficients, MSE, and deltas for a polynomial of degree n and other functions.
    Generate and save a plot for each function.
    Deltas are computed only for the polynomial.
    Returns a dictionary with function names, coefficients, MSE, and deltas for z.
    """
    n = len(x)
    results = {}

    # Compute means
    x_a = (x[0] + x[-1]) / 2  # Arithmetic mean
    x_g = np.sqrt(x[0] * x[-1])  # Geometric mean
    x_h = 2 / (1/x[0] + 1/x[-1])  # Harmonic mean
    y_a = (y[0] + y[-1]) / 2
    y_g = np.sqrt(y[0] * y[-1])
    y_h = 2 / (1/y[0] + 1/y[-1])

    # Compute polynomial of degree n (z)
    results['z'] = compute_polynomial(poly_degree, x, y, x_a, x_g, x_h, y_a, y_g, y_h)

    # Helper function to compute MSE
    def compute_mse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    # Helper function to create and save plot
    def create_plot(func_name, x, y, x_smooth, y_smooth):
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Data points')
        plt.plot(x_smooth, y_smooth, 'r-', label=f'Fitted curve: {func_name}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Least Squares Fit: {func_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{func_name}_plot.png')
        plt.close()

    # z1(x) = ax + b
    A1 = np.sum(x**2)
    B1 = np.sum(x)
    D1 = np.sum(x * y)
    E1 = np.sum(y)
    denom1 = A1 * n - B1**2
    a1 = (D1 * n - B1 * E1) / denom1
    b1 = (A1 * E1 - B1 * D1) / denom1
    y_pred1 = a1 * x + b1
    mse1 = compute_mse(y, y_pred1)
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth1 = a1 * x_smooth + b1
    create_plot('z1', x, y, x_smooth, y_smooth1)
    results['z1'] = {'a': a1, 'b': b1, 'mse': mse1}

    # z2(x) = ax^b -> ln y = ln a + b ln x
    ln_x = np.log(x)
    ln_y = np.log(y)
    A2 = np.sum(ln_x**2)
    B2 = np.sum(ln_x)
    D2 = np.sum(ln_x * ln_y)
    E2 = np.sum(ln_y)
    denom2 = A2 * n - B2**2
    b2 = (D2 * n - B2 * E2) / denom2
    ln_a2 = (A2 * E2 - B2 * D2) / denom2
    a2 = np.exp(ln_a2)
    y_pred2 = a2 * x**b2
    mse2 = compute_mse(y, y_pred2)
    y_smooth2 = a2 * x_smooth**b2
    create_plot('z2', x, y, x_smooth, y_smooth2)
    results['z2'] = {'a': a2, 'b': b2, 'mse': mse2}

    # z3(x) = ae^(bx) -> ln y = ln a + bx
    A3 = np.sum(x**2)
    B3 = np.sum(x)
    D3 = np.sum(x * ln_y)
    E3 = np.sum(ln_y)
    denom3 = A3 * n - B3**2
    b3 = (D3 * n - B3 * E3) / denom3
    ln_a3 = (A3 * E3 - B3 * D3) / denom3
    a3 = np.exp(ln_a3)
    y_pred3 = a3 * np.exp(b3 * x)
    mse3 = compute_mse(y, y_pred1)
    y_smooth3 = a3 * np.exp(b3 * x_smooth)
    create_plot('z3', x, y, x_smooth, y_smooth3)
    results['z3'] = {'a': a3, 'b': b3, 'mse': mse3}

    # z4(x) = a ln x + b
    A4 = np.sum(ln_x**2)
    B4 = np.sum(ln_x)
    D4 = np.sum(ln_x * y)
    E4 = np.sum(y)
    denom4 = A4 * n - B4**2
    a4 = (D4 * n - B4 * E4) / denom4
    b4 = (A4 * E4 - B4 * D4) / denom4
    y_pred4 = a4 * ln_x + b4
    mse4 = compute_mse(y, y_pred4)
    y_smooth4 = a4 * np.log(x_smooth) + b4
    create_plot('z4', x, y, x_smooth, y_smooth4)
    results['z4'] = {'a': a4, 'b': b4, 'mse': mse4}

    # z5(x) = a + b/x
    inv_x = 1 / x
    A5 = np.sum(inv_x**2)
    B5 = np.sum(inv_x)
    D5 = np.sum(inv_x * y)
    E5 = np.sum(y)
    denom5 = A5 * n - B5**2
    b5 = (D5 * n - B5 * E5) / denom5
    a5 = (A5 * E5 - B5 * D5) / denom5
    y_pred5 = a5 + b5 / x
    mse5 = compute_mse(y, y_pred5)
    y_smooth5 = a5 + b5 / x_smooth
    create_plot('z5', x, y, x_smooth, y_smooth5)
    results['z5'] = {'a': a5, 'b': b5, 'mse': mse5}

    # z6(x) = 1/(ax + b) -> 1/y = ax + b
    inv_y = 1 / y
    A6 = np.sum(x**2)
    B6 = np.sum(x)
    D6 = np.sum(x * inv_y)
    E6 = np.sum(inv_y)
    denom6 = A6 * n - B6**2
    a6 = (D6 * n - B6 * E6) / denom6
    b6 = (A6 * E6 - B6 * D6) / denom6
    y_pred6 = 1 / (a6 * x + b6)
    mse6 = compute_mse(y, y_pred6)
    y_smooth6 = 1 / (a6 * x_smooth + b6)
    create_plot('z6', x, y, x_smooth, y_smooth6)
    results['z6'] = {'a': a6, 'b': b6, 'mse': mse6}

    # z7(x) = x/(ax + b) -> 1/y = a + b/x
    A7 = np.sum(inv_x**2)
    B7 = np.sum(inv_x)
    D7 = np.sum(inv_x * inv_y)
    E7 = np.sum(inv_y)
    denom7 = A7 * n - B7**2
    b7 = (D7 * n - B7 * E7) / denom7
    a7 = (A7 * E7 - B7 * D7) / denom7
    y_pred7 = x / (a7 * x + b7)
    mse7 = compute_mse(y, y_pred7)
    y_smooth7 = x_smooth / (a7 * x_smooth + b7)
    create_plot('z7', x, y, x_smooth, y_smooth7)
    results['z7'] = {'a': a7, 'b': b7, 'mse': mse7}

    # z8(x) = ae^(b/x) -> ln y = ln a + b/x
    A8 = np.sum(inv_x**2)
    B8 = np.sum(inv_x)
    D8 = np.sum(inv_x * ln_y)
    E8 = np.sum(ln_y)
    denom8 = A8 * n - B8**2
    b8 = (D8 * n - B8 * E8) / denom8
    ln_a8 = (A8 * E8 - B8 * D8) / denom8
    a8 = np.exp(ln_a8)
    y_pred8 = a8 * np.exp(b8 / x)
    mse8 = compute_mse(y, y_pred8)
    y_smooth8 = a8 * np.exp(b8 / x_smooth)
    create_plot('z8', x, y, x_smooth, y_smooth8)
    results['z8'] = {'a': a8, 'b': b8, 'mse': mse8}

    # z9(x) = 1/(a ln x + b) -> 1/y = a ln x + b
    A9 = np.sum(ln_x**2)
    B9 = np.sum(ln_x)
    D9 = np.sum(ln_x * inv_y)
    E9 = np.sum(inv_y)
    denom9 = A9 * n - B9**2
    a9 = (D9 * n - B9 * E9) / denom9
    b9 = (A9 * E9 - B9 * D9) / denom9
    y_pred9 = 1 / (a9 * ln_x + b9)
    mse9 = compute_mse(y, y_pred9)
    y_smooth9 = 1 / (a9 * np.log(x_smooth) + b9)
    create_plot('z9', x, y, x_smooth, y_smooth9)
    results['z9'] = {'a': a9, 'b': b9, 'mse': mse9}

    return results, x_a, x_g, x_h, y_a, y_g, y_h

# Data
x = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
y = np.array([1.15, 1.39, 1.85, 1.95, 2.16, 2.79, 2.88, 2.38, 3.51])
poly_degree=3
# Compute coefficients, MSE, and deltas for polynomial of degree 4
results, x_a, x_g, x_h, y_a, y_g, y_h = compute_least_squares_coefficients(x, y, poly_degree)

# Print results
print(f"x_a = {x_a:.1f}, x_g = {x_g:.1f}, x_h = {x_h:.1f}")
print(f"y_a = {y_a:.1f}, y_g = {y_g:.1f}, y_h = {y_h:.1f}")
print("\nFunction coefficients, MSE, and deltas (for z only):")
for func, params in results.items():
    if func == 'z':
        coeff_str = ", ".join(f"a_{i} = {params[f'a_{i}']}" for i in range(poly_degree, -1, -1))
        print(f"\n{func}(x): {coeff_str}, MSE = {params['mse']}")
        for delta_name, delta_value in params['deltas'].items():
            print(f"  {delta_name} = {delta_value}")
    else:
        print(f"\n{func}(x): a = {params['a']}, b = {params['b']}, MSE = {params['mse']}")