import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate synthetic stock market onion price data
np.random.seed(42)
days = 1000
time = np.arange(0, days)

# Create synthetic onion prices (trend + noise)
prices = 50 + 5 * np.sin(0.02 * time) + np.random.normal(0, 0.5, days)
prices = prices.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Create sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(prices_scaled, sequence_length)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, scaler):
    y_true_rescaled = scaler.inverse_transform(y_true)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    r2 = r2_score(y_true_rescaled, y_pred_rescaled)
    mape = mean_absolute_percentage_error(y_true_rescaled, y_pred_rescaled)
    return r2, mape

# Define the model building function for Keras Tuner
def build_lstm_model(hp):
    model = Sequential()
    
    # Tune the number of LSTM layers
    for i in range(hp.Int("num_layers", 1, 3)):  # 1 to 3 LSTM layers
        units = hp.Int(f"units_{i}", min_value=32, max_value=128, step=32)
        model.add(LSTM(units, activation="tanh", return_sequences=(i < hp.get("num_layers") - 1)))
        model.add(Dropout(hp.Float(f"dropout_{i}", min_value=0.1, max_value=0.5, step=0.1)))
    
    # Tune activation function for the Dense layer
    activation = hp.Choice("activation", values=["relu", "tanh", "sigmoid"])
    model.add(Dense(1, activation=activation))
    
    # Tune learning rate
    learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    
    # Compile the model with the tuned learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model

# Create a Keras Tuner with RandomSearch (limit trials to 5)
tuner = kt.RandomSearch(
    build_lstm_model,
    objective="val_loss",
    max_trials=10,  # Run exactly 5 trials
    directory="lstm_tuning",
    project_name="stock_market"
)

# Callback to save hyperparameter combinations and metrics after each trial
results = []

class CSVLoggerCallback(tf.keras.callbacks.Callback):
    def on_trial_end(self, trial):
        hps = trial.hyperparameters.values
        val_loss = trial.score
        results.append({**hps, "val_loss": val_loss})

# Perform hyperparameter search
tuner.search(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=10, callbacks=[CSVLoggerCallback()])

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best number of layers: {best_hps.get('num_layers')}")
print(f"Best units in layer 0: {best_hps.get('units_0')}")
print(f"Best dropout in layer 0: {best_hps.get('dropout_0')}")
print(f"Best activation function: {best_hps.get('activation')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("hyperparameter_tuning_results.csv", index=False)

# Build the best model using the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
history = best_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# Predict on the test set using the tuned model
y_tuned_pred = best_model.predict(X_test)

# Evaluate the tuned model
loss, mae = best_model.evaluate(X_test, y_test)
print(f"Tuned Model - Test Loss: {loss}, Test MAE: {mae}")

# Calculate metrics for the tuned model
r2_tuned, mape_tuned = calculate_metrics(y_test, y_tuned_pred, scaler)
print(f"Tuned Model - R²: {r2_tuned:.4f}, MAPE: {mape_tuned:.4f}")

# Plot the results
y_pred_rescaled = scaler.inverse_transform(y_tuned_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label="Actual Prices")
plt.plot(y_pred_rescaled, label="Predicted Prices")
plt.legend()
plt.title("Actual vs Predicted Onion Prices")
plt.xlabel("Day")
plt.ylabel("Price")
plt.show()

