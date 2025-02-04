import matplotlib.pyplot as plt

# Example data
x = [0, 1, 2, 3, 4, 5]  # X-axis values
y = [0, 1, 4, 9, 16, 25]  # Y-axis values (e.g., square of x)

# Create the plot
plt.plot(x, y, label='y = x^2', color='blue', marker='o', linestyle='--')

# Add labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Line Plot Example")

# Add a legend
plt.legend()

# Display the grid
plt.grid(True)

# Show the plot
plt.show()

