import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from LNN_module import LiquidStateMachine
import pdb

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    num_samples = 1000
    sequence_length = 100
    input_size = 1
    liquid_size = 64
    output_size = 1

    X, y = generate_complex_data(num_samples, sequence_length)
    X = X.reshape(-1, sequence_length, input_size)
    y = y.reshape(-1, 1)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, input_size)).reshape(-1, sequence_length, input_size)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)

    X_train = X_scaled[:train_size]
    y_train = y_scaled[:train_size]
    X_val = X_scaled[train_size:train_size+val_size]
    y_val = y_scaled[train_size:train_size+val_size]
    X_test = X_scaled[train_size+val_size:]
    y_test = y_scaled[train_size+val_size:]

    model = LiquidStateMachine(
        input_size=input_size,
        liquid_size=liquid_size,
        output_size=output_size,
        spectral_radius=0.9,
        leak_rate=0.1
        )

    params = {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'patience': 10
        }

    history = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        learning_rate=params['learning_rate'],
        patience=params['patience']
        )

    # Make predictions
    train_predictions, _ = model(X_train, training=False)
    val_predictions, _ = model(X_val, training=False)
    test_predictions, _ = model(X_test, training=False)
    pdb.set_trace()
    # Inverse transform predictions
    train_predictions = scaler_y.inverse_transform(train_predictions[:, -1, :])
    val_predictions = scaler_y.inverse_transform(val_predictions[:, -1, :])
    test_predictions = scaler_y.inverse_transform(test_predictions[:, -1, :])

    y_train_orig = scaler_y.inverse_transform(y_train)
    y_val_orig = scaler_y.inverse_transform(y_val)
    y_test_orig = scaler_y.inverse_transform(y_test)

    # Calculate metrics
    train_mse = np.mean((train_predictions - y_train_orig) ** 2)
    val_mse = np.mean((val_predictions - y_val_orig) ** 2)
    test_mse = np.mean((test_predictions - y_test_orig) ** 2)

    print(f"Train MSE: {train_mse:.4f}")
    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Plot results
    plot_training_history(history)
    plot_predictions(y_test_orig, test_predictions, 'Test Set')

    # Plot example sequences
    X_test_orig = scaler_X.inverse_transform(X_test.reshape(-1, input_size)).reshape(-1, sequence_length, input_size)
    plot_example_sequences(X_test_orig, y_test_orig, test_predictions, sequence_length)


class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=10, min_delta=1e-4):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

def train_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=200, learning_rate=0.001, patience=10):
    """Utility function to train the model with proper callbacks"""

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    
    callbacks = [
        EarlyStopping(patience=patience),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
        ]
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
        )
    
    return history

def generate_complex_data(num_samples, sequence_length, noise_level=0.1):
    """
        Generates synthetic time series data combining multiple patterns:
        - Sine waves of different frequencies
        - Linear trends
        - Random walks
    """

    time = np.linspace(0, 4*np.pi, sequence_length)
    
    data = []
    labels = []
    for _ in range(num_samples):
        # Combine sine waves of different frequencies
        fast_sine = np.sin(time)
        slow_sine = 0.5 * np.sin(0.5 * time)
        
        # Add linear trend
        trend = 0.3 * np.linspace(-1, 1, sequence_length)
        
        # Add random walk component
        random_walk = np.cumsum(np.random.randn(sequence_length) * 0.1)
        
        # Combine components
        sequence = fast_sine + slow_sine + trend + random_walk
        
        # Add noise
        sequence += noise_level * np.random.randn(sequence_length)
        
        # Generate label (future value prediction)
        label = np.mean(sequence[-5:])  # Predict average of last 5 points
        
        data.append(sequence)
        labels.append(label)
    
    return np.array(data), np.array(labels)

# Modified training function without model saving
def train_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=200, learning_rate=0.001, patience=10):
    """Utility function to train the model with proper callbacks"""

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    
    callbacks = [
        EarlyStopping(patience=patience),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
        )
    
    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if 'lr' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(y_true, label='True Values', alpha=0.7)
    plt.plot(y_pred, label='Predictions', alpha=0.7)
    plt.title(f'{title} - True vs Predicted Values')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', alpha=0.8)
    plt.title(f'{title} - Prediction Scatter Plot')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.show()

# Plot example sequences with predictions
def plot_example_sequences(X, y_true, y_pred, sequence_length, num_examples=3):
    plt.figure(figsize=(15, 4 * num_examples))
    
    for i in range(num_examples):
        idx = np.random.randint(len(X))
        
        plt.subplot(num_examples, 1, i+1)
        plt.plot(X[idx, :, 0], label='Input Sequence', alpha=0.7)
        plt.scatter(sequence_length-1, y_true[idx], color='g', label='True Value', s=100)
        plt.scatter(sequence_length-1, y_pred[idx], color='r', label='Prediction', s=100)
        plt.title(f'Example Sequence {i+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
