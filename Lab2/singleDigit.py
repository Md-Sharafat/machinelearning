import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Display the fourth digit
plt.imshow(train_images[3], cmap='gray') 
plt.title(f'Label: {train_labels[3]}') 
plt.axis('off')  # Hide axis
plt.show()
