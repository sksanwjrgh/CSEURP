# 2-2) Root-finding methods [Newton's method]

# Define the function f(x)
# Define the derivative of the function f'(x)
def f(x):
    return x**5 - 9*x**4 - x**3 + 17*x**2 - 8*x - 8
def f_prime(x):
    return 5*x**4 - 36*x**3 - 3*x**2 + 34*x - 8

# Define the Newton's method function
def newton_method(f, f_prime, guess):
    count_newton = 0
    for i in range(1000):
        new_guess = guess - f(guess)/f_prime(guess)
        count_newton += 1
        if abs((new_guess-guess)/new_guess)<10**(-8):
            break
        else:
            guess = new_guess
    return new_guess, count_newton

# Execute Newton's method
root_newton, count_newton = newton_method(f, f_prime, -0.1)
print(f"The root of x0=-0.1 is: {root_newton:.8f}")
print(f"Number of iterations: {count_newton}")

root_newton, count_newton = newton_method(f, f_prime, -10)
print(f"The root of x0=-10 is: {root_newton:.8f}")
print(f"Number of iterations: {count_newton}")

root_newton, count_newton = newton_method(f, f_prime, 10)
print(f"The root of x0=10 is: {root_newton:.8f}")
print(f"Number of iterations: {count_newton}")