import tensorflow as tf

class LiquidStateMachine(tf.keras.Model):
    def __init__(self, input_size, liquid_size, output_size, spectral_radius=0.9):
        super(LiquidStateMachine, self).__init__()

        # Initialize the liquid layer
        self.liquid = tf.keras.layers.Dense(liquid_size, use_bias=False)
        liquid_weights = spectral_radius * tf.random.normal((input_size + liquid_size, liquid_size))
        self.liquid.kernel.assign(liquid_weights)

        # Initialize the readout layer
        self.readout = tf.keras.layers.Dense(output_size)

    def call(self, inputs, liquid_state=None):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]

        outputs = []

        # Initialize the liquid state if not provided
        if liquid_state is None:
            liquid_state = tf.zeros((batch_size, self.liquid.kernel.shape[1]))

        for t in range(time_steps):
            input_step = inputs[:, t, :]
            liquid_state = tf.tanh(self.liquid(tf.concat([input_step, liquid_state], axis=1)))
            output_step = self.readout(liquid_state)
            outputs.append(output_step)

        outputs = tf.stack(outputs, axis=1)
        return outputs, liquid_state