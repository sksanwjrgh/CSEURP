# Importing necessary libraries
import random
import numpy as np
import math

# 2-1) Root-finding methods [bisection method]
# Define the function f(x)
def f(x):
    return x**5 - 9*x**4 - x**3 + 17*x**2 - 8*x - 8
# Define the interval [a,b]

# Execute bisection method
def bisection_method(f, a, b):
    # define iteration count
    iteration_count = 0
    while abs(b-a)>10**(-8):
        m = abs(b-a)/2 + a
        if f(m) * f(a) < 0:
            b = m
        else:
            a = m
        iteration_count += 1
    return m,iteration_count

root, iteration_count = bisection_method(f, -10, -1)
print(f"The root of (-10,-1) is: {root}")
print(f"Number of iterations: {iteration_count}")

root, iteration_count = bisection_method(f, -1, 0)
print(f"The root of (-1,0) is: {root}")
print(f"Number of iterations: {iteration_count}")

root, iteration_count = bisection_method(f, 0, 10)
print(f"The root of (0,10) is: {root}")
print(f"Number of iterations: {iteration_count}")