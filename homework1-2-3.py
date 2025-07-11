# 2-3) Root-finding methods [Secant method]]
# Define the function f(x)
def f(x):
    return x**5 - 9*x**4 - x**3 + 17*x**2 - 8*x - 8
# Define the secant method function
def secant_method(f, x0, x1):
    count_secant = 0
    while abs((x1 - x0)/x1) > 10**(-8):
        if count_secant > 1000:
            print("Secant method did not converge.")
            return None, count_secant
        x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x_temp
        count_secant += 1
    return x1, count_secant
# Execute Secant method

root_secant, count_secant = secant_method(f, -10, -9.9)
print(f"The root of -10,-9,9 is: {root_secant}")
print(f"Number of iterations: {count_secant}")

root_secant, count_secant = secant_method(f, -0.1, -0.2)
print(f"The root of -0.1,-0.2 is: {root_secant}")
print(f"Number of iterations: {count_secant}")

root_secant, count_secant = secant_method(f, 10, 9.9)
print(f"The root of 10,9.9 is: {root_secant}")
print(f"Number of iterations: {count_secant}")