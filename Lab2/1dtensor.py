import numpy as np

# Define a 1D tensor
x = np.array([12, 3, 6, 14])
print("Print the elements of 1D tensor:", x)
print("Dimension of the tensor:", x.ndim)
print("Shape of the tensor:", x.shape)

# Define a 2D tensor
x1 = np.array([[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]])
print("Print the elements of 2D tensor:", x1)
print("Dimension of the tensor:", x1.ndim)
print("Shape of the tensor:", x1.shape)

# Define a 3D tensor
x2 = np.array([[[5, 78, 2, 34, 0],
                [6, 79, 3, 35, 1],
                [7, 80, 4, 36, 2]],
               [[5, 78, 2, 34, 0],
                [6, 79, 3, 35, 1],
                [7, 80, 4, 36, 2]],
               [[5, 78, 2, 34, 0],
                [6, 79, 3, 35, 1],
                [7, 80, 4, 36, 2]]])
print("Print the elements of 3D tensor:", x2)
print("Dimension of the tensor:", x2.ndim)
print("Shape of the tensor:", x2.shape)

