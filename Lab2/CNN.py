import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define the CNN Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

file_path = '/home/srabanti/Downloads/Electric_Production.csv'
data = pd.read_csv(file_path)

print(data)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data['IPG2211A2N_scaled'] = scaler.fit_transform(data[['IPG2211A2N']])

# Create sequences for the CNN model
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Define the sequence length
sequence_length = 30 

# Prepare the data for supervised learning
data_values = data['IPG2211A2N_scaled'].values
X, y = create_sequences(data_values, sequence_length)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Print shapes of the datasets
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Reshape data for the CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1)  # Output layer for regression
])

# Compile the Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform predictions
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Inverse transform actual values
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Create a DataFrame to compare predictions and actual values
print("actual and predicted value for training data ")
training_table = pd.DataFrame({
    'Actual Values': y_train_inv.flatten(),
    'Predicted Values': train_predictions.flatten()
})


print(training_table)

print("actual and predicted value for testing data ")
testing_table = pd.DataFrame({
    'Actual Values': y_train_inv.flatten(),
    'Predicted Values': train_predictions.flatten()
})

print(testing_table)

# Evaluate the Model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss value is: {loss:.4f}, MAE value is: {mae:.4f}")

# Create a figure with 2 subplots (1 row, 2 columns)
plt.figure(figsize=(14, 7))

# Plot 1: Training Data vs Predictions
plt.subplot(1, 2, 1)  
plt.plot(y_train_inv, label='Actual Training Data', color='blue')
plt.plot(
    np.arange(sequence_length, len(train_predictions) + sequence_length),
    train_predictions,
    label='Predicted Training Data',
    color='orange'
)
plt.title('Training Data vs Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Electric Production')
plt.legend()
plt.grid()

# Plot 2: Testing Data vs Predictions
plt.subplot(1, 2, 2) 
plt.plot(y_test_inv, label='Actual Test Data', color='green')
plt.plot(
    np.arange(len(test_predictions)),
    test_predictions,
    label='Predicted Test Data',
    color='red'
)
plt.title('Test Data vs Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Electric Production')
plt.legend()
plt.grid()

# Show the combined plot
plt.tight_layout()  
plt.show()

