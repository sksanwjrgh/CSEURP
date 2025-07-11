import numpy as np
import matplotlib.pyplot as plt

# Defining f(x)
def f(x):
    return 1/(1+16*x**2)
x_nodes = np.arange(-1,1.2,0.2)
y_nodes = f(x_nodes)

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
        p += y_nodes[i]*lagrange_basis(x,x_nodes,i)
    return p

# Executing p(x) and f(x)
x_values = np.linspace(-1,1,1000)
p_values = np.array([lagrange_interpolation(x,x_nodes) for x in x_values])
f_values = f(x_values)

# plotting
plt.plot(x_values,f_values,label="f(x)",color="blue")
plt.plot(x_values,p_values,label="Lagrange Interpolation",color="red")
plt.scatter(x_nodes, y_nodes, color='green', zorder=5, label="Interpolation Nodes")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-0.1, 1.4)
plt.grid(True)
plt.legend()
plt.show()


        
        

    