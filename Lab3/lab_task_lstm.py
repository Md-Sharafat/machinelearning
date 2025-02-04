import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt
import tensorflow as tf

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

# Define the model building function for Keras Tuner
def build_lstm_model(hp):
    model = Sequential()
    
    for i in range(hp.Int("num_layers", 1, 3)):  # 1 to 3 LSTM layers
        units = hp.Int(f"units_{i}", min_value=32, max_value=128, step=32)
        model.add(LSTM(units, activation="tanh", return_sequences=(i < hp.get("num_layers") - 1)))
        model.add(Dropout(hp.Float(f"dropout_{i}", min_value=0.1, max_value=0.5, step=0.1)))
    
    activation = hp.Choice("activation", values=["relu", "tanh", "sigmoid"])
    model.add(Dense(1, activation=activation))
    
    learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse", metrics=["mae"])
    return model

# Create a Keras Tuner with Hyperband
tuner = kt.Hyperband(
    build_lstm_model,
    objective="val_loss",
    max_epochs=30,
    factor=3,
    directory="lstm_tuning",
    project_name="stock_market"
)

# Perform hyperparameter search
tuner.search(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=30)

# Get all trials
all_trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))

# Get the best hyperparameters and build the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Prepare trial results
results = []
for trial in all_trials:
    trial_result = {
        "trial_id": trial.trial_id,
        "num_layers": trial.hyperparameters.get("num_layers"),
        "best_num_layers": best_hps.get("num_layers"),
        "units_0": trial.hyperparameters.get("units_0"),
        "best_units_0": best_hps.get("units_0"),
        "dropout_0": trial.hyperparameters.get("dropout_0"),
        "best_dropout_0": best_hps.get("dropout_0"),
        "activation": trial.hyperparameters.get("activation"),
        "best_activation": best_hps.get("activation"),
        "learning_rate": trial.hyperparameters.get("learning_rate"),
        "best_learning_rate": best_hps.get("learning_rate"),
        "val_loss": trial.score,
        "best_val_loss": min(history.history["val_loss"]),
        # Get 'mae' from the history object instead of trial.metrics
        "val_mae": trial.metrics.get_history('mae')[-1],  # Get the last recorded 'mae' value
        "best_val_mae": min(history.history["val_mae"])
    }
    # ... (rest of the code remains the same)
    
    # Add units and dropout for additional layers if they exist
    if trial.hyperparameters.get("num_layers") >= 2:
        trial_result["units_1"] = trial.hyperparameters.get("units_1")
        trial_result["dropout_1"] = trial.hyperparameters.get("dropout_1")
    if trial.hyperparameters.get("num_layers") == 3:
        trial_result["units_2"] = trial.hyperparameters.get("units_2")
        trial_result["dropout_2"] = trial.hyperparameters.get("dropout_2")

    if best_hps.get("num_layers") >= 2:
        trial_result["best_units_1"] = best_hps.get("units_1")
        trial_result["best_dropout_1"] = best_hps.get("dropout_1")
    if best_hps.get("num_layers") == 3:
       trial_result["best_units_2"] = best_hps.get("units_2")
       trial_result["best_dropout_2"] = best_hps.get("dropout_2")
    
        
    results.append(trial_result)


# Add the best hyperparameters as the last row
best_result = {
    "trial_id": "best_model_hyperParameter",
    "num_layers": best_hps.get("num_layers"),
    "units_0": best_hps.get("units_0"),
    "dropout_0": best_hps.get("dropout_0"),
    "activation": best_hps.get("activation"),
    "learning_rate": best_hps.get("learning_rate"),
    "val_loss": min(history.history["val_loss"]),
}

if best_hps.get("num_layers") >= 2:
    best_result["units_1"] = best_hps.get("units_1")
    best_result["dropout_1"] = best_hps.get("dropout_1")
if best_hps.get("num_layers") == 3:
    best_result["units_2"] = best_hps.get("units_2")
    best_result["dropout_2"] = best_hps.get("dropout_2")

results.append(best_result)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("B190305045.csv", index=False)

print("Results for all trials and best model saved to 'B190305045(nusrat_alam_srabanti).csv'")

