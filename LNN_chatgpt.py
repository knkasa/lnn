import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the LiquidStateMachine class (from the revised code above)
class LiquidStateMachine(tf.keras.Model):
    def __init__(self, input_size, liquid_size, output_size, spectral_radius=0.9):
        super(LiquidStateMachine, self).__init__()
        
        # Initialize the liquid layer
        self.liquid = tf.keras.layers.Dense(
            liquid_size, use_bias=False, activation=None
        )
        self.input_size = input_size
        self.liquid_size = liquid_size
        
        # Initialize the readout layer
        self.readout = tf.keras.layers.Dense(output_size)

        # Set custom weights for the liquid layer with spectral radius scaling
        initial_weights = tf.random.normal((input_size + liquid_size, liquid_size))
        eigenvalues, _ = tf.linalg.eigh(tf.matmul(initial_weights, initial_weights, transpose_b=True))
        max_eigenvalue = tf.reduce_max(eigenvalues)
        scaled_weights = initial_weights * (spectral_radius / max_eigenvalue)
        
        # Assign the scaled weights after the layer has been built
        self.liquid.build((None, input_size + liquid_size))
        self.liquid.set_weights([scaled_weights])

    def call(self, inputs, liquid_state=None):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        # TensorArray to store the outputs at each time step
        output_array = tf.TensorArray(dtype=tf.float32, size=time_steps)

        # Initialize the liquid state if not provided
        if liquid_state is None:
            liquid_state = tf.zeros((batch_size, self.liquid_size))

        # Iterate through the time steps
        for t in tf.range(time_steps):
            input_step = inputs[:, t, :]
            # Concatenate the input and liquid state
            combined_input = tf.concat([input_step, liquid_state], axis=1)
            liquid_state = tf.tanh(self.liquid(combined_input))
            output_step = self.readout(liquid_state)
            output_array = output_array.write(t, output_step)

        # Stack the outputs and transpose to match the expected shape
        outputs = tf.transpose(output_array.stack(), perm=[1, 0, 2])
        return outputs, liquid_state


# Generate Sample Data
def generate_synthetic_data(sequence_length, num_samples):
    """Generates synthetic time series data for regression."""
    x = np.linspace(0, 2 * np.pi, sequence_length)
    y = np.sin(x)  # Sine wave as target
    data = np.array([y + 0.1 * np.random.randn(sequence_length) for _ in range(num_samples)])
    labels = np.sum(data, axis=1, keepdims=True)  # Sum of sequence as target label
    return data, labels

# Hyperparameters
input_size = 1  # One feature
sequence_length = 50
num_samples = 1000
liquid_size = 16
output_size = 1
epochs = 200
batch_size = 32
learning_rate = 0.001

# Generate data
data, labels = generate_synthetic_data(sequence_length, num_samples)
data = data[..., np.newaxis]  # Add a feature dimension
train_size = int(0.8 * num_samples)

x_train, y_train = data[:train_size], labels[:train_size]
x_test, y_test = data[train_size:], labels[train_size:]

# Build and compile the model
model = LiquidStateMachine(input_size=input_size, liquid_size=liquid_size, output_size=output_size)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.MeanSquaredError())

# Train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Evaluate the model
test_loss = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss[0]:.4f}")

# Make predictions
predictions, _ = model(x_test)
actual_preds = predictions[:,-1,:]