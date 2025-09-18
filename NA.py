import tensorflow as tf


def layer_norm(x, epsilon=1e-5, scope='layer_norm'):
    with tf.variable_scope(scope):
        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        scale = tf.get_variable('scale', shape=[x.get_shape()[-1]], initializer=tf.ones_initializer())
        shift = tf.get_variable('shift', shape=[x.get_shape()[-1]], initializer=tf.zeros_initializer())
        normalized = (x - mean) / tf.sqrt(variance + epsilon)

        return scale * normalized + shift


class VisualNoiseAwareModule(object):
    def __init__(self, dim, name="", dropout_rate=0.1, depth=3):
        self.dim = dim
        self.name = name
        self.dropout_rate = dropout_rate
        self.depth = depth
        with tf.variable_scope(name):
            self.threshold_bias = tf.get_variable(
                name='{}_threshold_bias'.format(self.name),
                shape=[1, self.dim],
                initializer=tf.zeros_initializer(),
                trainable=True
            )

    def __call__(self, inputs, training=True):
        with tf.variable_scope("VisualNoiseAwareModule_{}".format(self.name), reuse=tf.AUTO_REUSE):
            hidden = inputs
            for i in range(self.depth):
                hidden = tf.layers.dense(hidden, self.dim * 2, activation=tf.nn.relu, name=f"dense_{i}")
                hidden = tf.layers.dropout(hidden, rate=self.dropout_rate, training=training, name=f"dropout_{i}")

            hidden = tf.layers.dense(hidden, self.dim, activation=tf.nn.relu, name="dense_out")
            hidden = layer_norm(hidden, scope="layer_norm")
            noise_level = tf.layers.dense(hidden, self.dim, activation=tf.nn.sigmoid, name="noise_level")
            dynamic_weight = 1.0 - noise_level

            return dynamic_weight


class TextualNoiseAwareModule(object):
    def __init__(self, dim, name="", dropout_rate=0.1, num_heads=2):
        self.dim = dim
        self.name = name
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        with tf.variable_scope(name):
            self.threshold_bias = tf.get_variable(
                name='{}_threshold_bias'.format(self.name),
                shape=[1, self.dim],
                initializer=tf.zeros_initializer(),
                trainable=True
            )

    def __call__(self, inputs, training=True):
        with tf.variable_scope("TextualNoiseAwareModule_{}".format(self.name), reuse=tf.AUTO_REUSE):
            hidden = tf.layers.dense(inputs, self.dim, activation=tf.nn.relu, name="proj_in")
            hidden = tf.layers.dropout(hidden, rate=self.dropout_rate, training=training)
            q = tf.layers.dense(hidden, self.dim, name="q")
            k = tf.layers.dense(hidden, self.dim, name="k")
            v = tf.layers.dense(hidden, self.dim, name="v")
            attn_scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.dim))
            attn_weights = tf.nn.softmax(attn_scores)
            attn_out = tf.matmul(attn_weights, v)
            merged = tf.layers.dense(attn_out, self.dim, activation=tf.nn.relu)
            merged = layer_norm(merged, scope="layer_norm")
            noise_level = tf.layers.dense(merged, self.dim, activation=tf.nn.sigmoid)
            dynamic_weight = 1.0 - noise_level

            return dynamic_weight