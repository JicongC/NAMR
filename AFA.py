import tensorflow as tf
from tensorflow.keras import layers

class AdaptiveGlobalLocalAttention(layers.Layer):
    def __init__(self, units, activation='sigmoid', dropout_rate=0.2, use_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_scale = use_scale

    def build(self, input_shapes):
        shapes = input_shapes if isinstance(input_shapes[0], (list, tuple)) else [input_shapes, input_shapes]
        shp = shapes[0]
        nd = len(shp)
        if nd == 3:
            Conv = layers.Conv1D
            GMP  = layers.GlobalMaxPooling1D
            GAP  = layers.GlobalAveragePooling1D
            self.pool_axes = [1]
            self.expand_spatial = False
        elif nd == 4:
            Conv = layers.Conv2D
            GMP  = layers.GlobalMaxPooling2D
            GAP  = layers.GlobalAveragePooling2D
            self.pool_axes = [1, 2]
            self.expand_spatial = True
        else:
            raise ValueError(f"Unsupported rank: {nd}")

        self.gmin1 = layers.Lambda(lambda x: tf.reduce_min(x, axis=self.pool_axes))
        self.gavg1 = GAP()
        self.gmax1 = GMP()
        self.gmin2 = layers.Lambda(lambda x: tf.reduce_min(x, axis=self.pool_axes))
        self.gavg2 = GAP()
        self.gmax2 = GMP()
        self.dense_global = layers.Dense(self.units, activation=self.activation)
        self.conv_l1 = Conv(self.units, kernel_size=1, activation=self.activation)
        self.conv_l2 = Conv(self.units, kernel_size=1, activation=self.activation)
        if self.use_scale:
            self.global_scale = self.add_weight("gs", shape=(1,), initializer="ones", trainable=True)
            self.local_scale  = self.add_weight("ls", shape=(self.units,), initializer="ones", trainable=True)

        super().build(input_shapes)

    def _make_spatial(self, x):
        shape = (-1, *([1]* (2 if self.expand_spatial else 1)), self.units)

        return tf.reshape(x, shape)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x1, x2 = inputs
        else:
            x1 = x2 = inputs

        stats1 = [self.gmin1(x1), self.gavg1(x1), self.gmax1(x1)]
        stats2 = [self.gmin2(x2), self.gavg2(x2), self.gmax2(x2)]
        fused = [self.dense_global(a + b) for a, b in zip(stats1, stats2)]
        global_vec = tf.add_n(fused)
        global_map = self._make_spatial(global_vec)
        l1 = self.conv_l1(x1)
        l2 = self.conv_l2(x2)
        for ax in self.pool_axes:
            l1 = tf.reduce_min(l1, axis=ax, keepdims=True), tf.reduce_mean(l1, axis=ax, keepdims=True), tf.reduce_max(l1, axis=ax, keepdims=True)
            l2 = tf.reduce_min(l2, axis=ax, keepdims=True), tf.reduce_mean(l2, axis=ax, keepdims=True), tf.reduce_max(l2, axis=ax, keepdims=True)

        locals1 = tf.concat(l1, axis=-1)
        locals2 = tf.concat(l2, axis=-1)
        local_map = locals1 + locals2
        local_map = layers.Conv2D(self.units, 1, activation=self.activation)(local_map) \
                    if self.expand_spatial else layers.Conv1D(self.units, 1, activation=self.activation)(local_map)

        if self.use_scale:
            global_map *= self.global_scale
            local_map  *= tf.reshape(self.local_scale, [*( [1]* ( (3 if self.expand_spatial else 2) ) ), self.units])

        att = tf.sigmoid(global_map + local_map)

        return att

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_scale": self.use_scale
        })

        return cfg


class AdaptiveAttentionLayer(layers.Layer):
    def __init__(self, units=64, use_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_scale = use_scale

    def build(self, input_shapes):
        if isinstance(input_shapes, (list, tuple)) and len(input_shapes) == 2:
            shapes = list(input_shapes)
        else:
            shapes = [input_shapes, input_shapes]

        shp1 = tf.TensorShape(shapes[0])
        ndims = len(shp1)
        C1 = int(shp1[-1])
        alpha_shape = [1] * (ndims - 1) + [C1]
        self.alpha1 = self.add_weight(
            name="alpha1",
            shape=alpha_shape,
            initializer="ones",
            trainable=True
        )
        shp2 = tf.TensorShape(shapes[1])
        C2 = int(shp2[-1])
        alpha2_shape = [1] * (ndims - 1) + [C2]
        self.alpha2 = self.add_weight(
            name="alpha2",
            shape=alpha2_shape,
            initializer="ones",
            trainable=True
        )
        self.agl = AdaptiveGlobalLocalAttention(self.units, use_scale=self.use_scale)
        if ndims == 3:
            dummy_shape = (None, None, C1)
        else:
            dummy_shape = (None, None, None, C1)

        self.agl.build([dummy_shape, dummy_shape])
        super().build(input_shapes)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x1, x2 = inputs
        else:
            x1 = x2 = inputs

        att = self.agl([x1, x2])
        out1 = x1 * att * self.alpha1
        out2 = x2 * att * self.alpha2

        return (out1, out2) if isinstance(inputs, (list, tuple)) else out1

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "use_scale": self.use_scale})

        return cfg

