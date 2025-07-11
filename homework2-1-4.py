import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import solve

#Defining f(x)
def f(x):
    return 1/(1+16*x**2)

#Defining x_nodes (cheby, uniform)
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

#Defining cubic_spline_interpol
def cubic_spline(x,x_nodes,y_nodes):
    # Define x_nodes's len
    n = len(x_nodes)
    
    # Define x_nodes, y_nodes's interval
    h = np.diff(x_nodes)

    # Define A vector (Ax=b)
    A = np.zeros((n-2,n-2))
    b = np.zeros(n-2)


    # Defining coefficients
    for i in range(1,n-1):
        A[i-1,i-1]=(h[i]+h[i-1])/3
        if i>1:
            A[i-1,i-2]=h[i-1]/6
        if i<n-2:
            A[i-1,i]=h[i]/6
        b[i-1]=(y_nodes[i+1]-y_nodes[i])/h[i]-(y_nodes[i]-y_nodes[i-1])/h[i-1]
    M_solution = solve(A,b)
    M = np.zeros(n)
    M[1:-1]=M_solution
    
    a_i= y_nodes
    b_i = []
    c_i = []
    d_i = []
    S = []
    x_full=[]
    
    # Defining S_x
    for i in range(n-1):
        
        # appending coefficients
        d_i.append((M[i+1]-M[i])/(6*h[i]))
        c_i.append(M[i]/2)
        b_i.append((y_nodes[i+1]-y_nodes[i])/h[i]-h[i]*(M[i+1] + 2*M[i]) / 6)
        
        x_local = np.linspace(x_nodes[i],x_nodes[i+1],100)
        t = x_local-x_nodes[i]
        s_i = a_i[i]+b_i[i]*t+c_i[i]*t**2+d_i[i]*t**3
        
        #appending S and x
        S.append(s_i)
        x_full.append(x_local)

    
    # Contiuning
    S = np.concatenate(S)
    return S, np.concatenate(x_full)

result_cheby = cubic_spline(x,x_nodes_cheby,y_nodes_cheby)
result_uniform = cubic_spline(x,x_nodes_uniform,y_nodes_uniform)
x_plot = np.linspace(-1, 1, 100)
y_lagrange_cheby = np.array([lagrange_interpolation(xi, x_nodes_cheby) for xi in x_plot])

#Plotting
plt.figure(figsize=(10,6))

plt.plot(np.linspace(-1, 1, 100), f(np.linspace(-1, 1, 100)),':', label='Original f(x)' ,color='black')
plt.plot(result_cheby[1],result_cheby[0],label='Cheby node(spline)',color="blue")
plt.plot(x_plot,y_lagrange_cheby,label='Cheby node(Lagrangian)',color="red")
 
""" plt.plot(result_uniform[1],result_uniform[0],label='Uniform node(spline)',color="green")
 """
plt.legend()

plt.grid(True)
plt.show()


        
