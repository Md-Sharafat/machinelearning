import numpy as np
from keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Select digits #10 to #100
my_slice = train_images[10:100]
print(my_slice.shape)  # Output: (90, 28, 28)
# Equivalent slicing
my_slice_1 = train_images[10:100, :, ] # All rows and all columns
print(my_slice_1.shape)  # Output: (90, 28, 28)

my_slice_2 = train_images[10:100, 0:28, 0:28]  # Specific range on all axes
print(my_slice_2.shape)  # Output: (90, 28, 28)
my_slice_bottom_right = train_images[:, 14:, 14:]  # Bottom-right 14x14 pixels
print(my_slice_bottom_right.shape)  # Shape will be (60000, 14, 14)
my_slice_centered = train_images[:, 7:-7, 7:-7]  # Centered 14x14 pixels
print(my_slice_centered.shape)  # Shape will be (60000, 14, 14)
# First batch of size 128
batch_1 = train_images[:128]
print(batch_1.shape)  # Output: (128, 28, 28)

# Second batch of size 128
batch_2 = train_images[128:256]
print(batch_2.shape)  # Output: (128, 28, 28)
n = 1  # Example for the second batch
batch_n = train_images[128 * n:128 * (n + 1)]
print(batch_n.shape)  # Output: (128, 28, 28)



