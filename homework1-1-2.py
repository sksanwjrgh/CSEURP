# 1) Matrix operations and Gauss-Jordan elimination
# Importing necessary libraries
import random
import numpy as np
import math

# Generating 2 4x4 matrix with random integers between 1 and 10000
matrixA = np.random.randint(1, 100,(4,4))
matrixB = np.random.randint(1, 100,(4,4))
matrixC = matrixA @ matrixB

# Print the matrix1 X matrix2
print(matrixC)


# define gauss_jordan_elimination function
# [A | I] extention
def gauss_jordan_elimination(matrix):
    rows, cols = matrix.shape
    augmented_matrix = np.hstack((matrix, np.identity(rows))) 

# if the diagonal element is zero, swap with a row below
    for i in range(rows):
        if augmented_matrix[i][i] == 0:
            for j in range(i+1, rows):
                if augmented_matrix[j][i] != 0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    break
        
        # To make diagonal element 1
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i][i]
    
        # making other elements in the column 0
        for j in range(rows):
            if j != i:
                augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j][i]    
    # return right part of the augmented matrix
    return augmented_matrix[:, rows:]

# calculating the inverse of matrixC using function
C_inv = gauss_jordan_elimination(matrixC)
print("\nThe inverse matrix is:")
print(C_inv)

# checking if the product of matrixC and its inverse gives the identity matrix
identity_matrix_4x4 = matrixC@C_inv
print(identity_matrix_4x4)