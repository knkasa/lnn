import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class LiquidStateMachine(tf.keras.Model):
    def __init__(self, input_size, liquid_size, output_size, spectral_radius=0.9, 
                 leak_rate=0.1, activation='tanh', readout_activation='linear'):
        super(LiquidStateMachine, self).__init__()
        
        self.input_size = input_size
        self.liquid_size = liquid_size
        self.leak_rate = leak_rate
        
        self.liquid = tf.keras.layers.Dense(
            liquid_size,
            use_bias=True,  # Added bias for better expressiveness
            activation=None,
            kernel_initializer='glorot_uniform'
            )
        
        self.batch_norm = tf.keras.layers.BatchNormalization()
        
        # Multiple readout layers for better feature extraction
        self.readout_layers = [
            tf.keras.layers.Dense(liquid_size // 2, activation='relu'),
            tf.keras.layers.Dense(liquid_size // 4, activation='relu'),
            tf.keras.layers.Dense(output_size, activation=readout_activation)
            ]
        
        self.dropout = tf.keras.layers.Dropout(0.2)
        
        # Initialize weights with spectral radius scaling
        self._initialize_weights(spectral_radius)
        
        # Store activation function
        self.activation_fn = tf.keras.activations.get(activation)

    def _initialize_weights(self, spectral_radius):
        initial_weights = tf.random.normal((self.input_size + self.liquid_size, self.liquid_size))

        # Use singular value decomposition for more stable spectral radius scaling
        s, u, v = tf.linalg.svd(initial_weights)
        scaled_s = s * (spectral_radius / tf.reduce_max(s))
        scaled_weights = tf.matmul(u, tf.matmul(tf.linalg.diag(scaled_s), v, adjoint_b=True))
        
        # Build and set weights
        self.liquid.build((None, self.input_size + self.liquid_size))
        self.liquid.set_weights([scaled_weights, tf.zeros(self.liquid_size)])

    @tf.function
    def call(self, inputs, liquid_state=None, training=False):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        output_array = tf.TensorArray(dtype=tf.float32, size=time_steps)
        
        if liquid_state is None:
            liquid_state = tf.zeros((batch_size, self.liquid_size))
        
        for t in tf.range(time_steps):
            input_step = inputs[:, t, :]
            combined_input = tf.concat([input_step, liquid_state], axis=1)
            
            # Apply liquid dynamics with leaky integration
            liquid_input = self.liquid(combined_input)
            liquid_input = self.batch_norm(liquid_input, training=training)
            liquid_state = (1 - self.leak_rate) * liquid_state + \
                          self.leak_rate * self.activation_fn(liquid_input)
            
            if training:
                liquid_state = self.dropout(liquid_state)
            
            # Process through readout layers
            output_step = liquid_state
            for layer in self.readout_layers:
                output_step = layer(output_step)
            
            output_array = output_array.write(t, output_step)

        outputs = tf.transpose(output_array.stack(), perm=[1, 0, 2])

        return outputs, liquid_state
