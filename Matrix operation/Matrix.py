import numpy as np

A = np.array([[2, 3, 4],
              [1, 0, 6],
              [7, 5, 9]])

B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("Matrix A:\n", A)
print("Matrix B:\n", B)

add = np.add(A, B)
print("\nMatrix Addition:\n", add)

multiply = np.dot(A, B)
print("\nMatrix Multiplication:\n", multiply)

transpose_A = np.transpose(A)
print("\nTranspose of Matrix A:\n", transpose_A)

det_A = np.linalg.det(A)
print("\nDeterminant of Matrix A:", det_A)
