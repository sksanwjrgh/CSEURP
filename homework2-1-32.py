import numpy as np
import matplotlib.pyplot as plt
import math

# Defining f(x)
def f(x):
    return 1/(1+16*x**2)

# Defining x_nodes (cheby, uniform)
x_nodes_cheby = []
for x in range(11):
    x_cheby = math.cos((2*x+1)/(24)*math.pi)
    x_nodes_cheby.append(x_cheby)
x_nodes_cheby = np.array(x_nodes_cheby)
x_nodes_uniform = np.arange(-1,1.2,0.2)

y_nodes_cheby = f(x_nodes_cheby)
y_nodes_uniform = f(x_nodes_uniform)


# Defining lagrange_basis
def lagrange_basis(x,x_nodes,i):
    count = len(x_nodes)
    L=1
    for j in range(count):
        if j!=i:
            L = L * (x-x_nodes[j])/(x_nodes[i]-x_nodes[j])
    return L

# Defining lagrange_interpolation
def lagrange_interpolation(x,x_nodes):
    p = 0
    for i in range(len(x_nodes)):
        p += f(x_nodes[i])*lagrange_basis(x,x_nodes,i)
    return p

# Defining x-x_i
def difference(x,x_nodes):
    l = 1
    for i in range(11):
        l = l*(x-x_nodes[i])
    return l

# Executing p(x)_cheby and p(x)_uniform and f(x)
x_values = np.linspace(-1,1,1000)
f_values = f(x_values)

p_values_cheby = np.array([lagrange_interpolation(x,x_nodes_cheby) for x in x_values])
p_values_uniform = np.array([lagrange_interpolation(x,x_nodes_uniform) for x in x_values])
difference_uniform = np.array([difference(x,x_nodes_uniform) for x in x_values])
difference_cheby = np.array([difference(x,x_nodes_cheby) for x in x_values])

# plotting (comparing Cheby and uniform method)
plt.plot(x_values,difference_uniform,label="Uniform difference",color="yellow")
plt.plot(x_values,difference_cheby,label="cheby difference",color="black")

plt.xlabel("x")
plt.ylabel("y")
plt.ylim(0, 0.009)
plt.grid(True)
plt.legend()
plt.show()


        
        

    