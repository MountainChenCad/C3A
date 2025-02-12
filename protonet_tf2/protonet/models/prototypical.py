import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, MaxPool2D, Add, GlobalAveragePooling2D, LeakyReLU, Layer
from tensorflow.keras import Model

def calc_euclidian_dists(x, y):
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)

class DropBlock2D(Layer):
    def __init__(self, keep_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def call(self, inputs, training=False):
        if not training:
            return inputs
        # Compute the mask
        gamma = (1. - self.keep_prob) * tf.math.reduce_prod(tf.cast(inputs.shape[1:3], tf.float32)) / (self.block_size ** 2)
        mask = tf.cast(tf.random.uniform(tf.shape(inputs)) < gamma, tf.float32)
        mask = tf.nn.max_pool2d(mask, ksize=self.block_size, strides=1, padding='SAME')
        mask = 1 - mask
        return inputs * mask

    def get_config(self):
        config = super(DropBlock2D, self).get_config()
        config.update({
            'keep_prob': self.keep_prob,
            'block_size': self.block_size,
        })
        return config

    def compute_output_shape(self, input_shape):
        # 输出形状与输入形状相同
        return input_shape

class ResidualBlock(Layer):
    def __init__(self, filters, downsample=False):
        super(ResidualBlock, self).__init__()
        self.filters = filters  # 保存 filters 参数
        self.downsample = downsample
        self.conv1 = Conv2D(filters, kernel_size=3, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = LeakyReLU()
        self.conv2 = Conv2D(filters, kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = LeakyReLU()
        self.conv3 = Conv2D(filters, kernel_size=3, padding='same')
        self.bn3 = BatchNormalization()
        self.downsample_conv = Conv2D(filters, kernel_size=1, padding='valid')
        self.bn_downsample = BatchNormalization()
        self.maxpool = MaxPool2D((2, 2))
        self.dropblock = DropBlock2D(keep_prob=0.5, block_size=5)

    def call(self, inputs):
        # print("ResidualBlock input shape:", inputs.shape)
        shortcut = self.downsample_conv(inputs)
        shortcut = self.bn_downsample(shortcut)

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += shortcut
        x = self.relu2(x)

        if self.downsample:
            x = self.maxpool(x)
            x = self.dropblock(x)

        # print("ResidualBlock output shape:", x.shape)
        return x

    def compute_output_shape(self, input_shape):
        if self.downsample:
            # 如果有下采样（最大池化），空间维度减半
            return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, self.filters)
        else:
            # 如果没有下采样，空间维度不变，通道数变为 filters
            return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'downsample': self.downsample,
        })
        return config

class Prototypical(Model):
    def __init__(self, n_support, n_query, w, h, c, encoder_type='resnet12', nb_layers=4, nb_filters=64, base_model=None):
        super(Prototypical, self).__init__()
        self.w, self.h, self.c = w, h, c

        layers = []
        if base_model is not None:
            layers.append(base_model)

        if encoder_type == 'conv64F':
            for i in range(nb_layers):
                layers += self.conv_block(nb_filters=nb_filters)
            layers.append(Flatten())
            self.encoder = tf.keras.Sequential(layers)
        elif encoder_type == 'resnet12':
            filters = [64, 160, 320, 640]  # 每个残差块的 filters 数量
            for i in range(nb_layers):
                layers.append(ResidualBlock(filters[i], downsample=True))
            layers.append(GlobalAveragePooling2D())
            layers.append(DropBlock2D(keep_prob=0.5, block_size=5))
            layers.append(Flatten())
            self.encoder = tf.keras.Sequential(layers)
        else:
            raise ValueError("Unsupported encoder type. Choose 'conv64F' or 'resnet12'.")

    def conv_block(self, kernel_size=3, padding='same', nb_filters=None):
        return [
            Conv2D(filters=nb_filters, kernel_size=kernel_size, padding=padding),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2))
        ]

    def call(self, support, query):
        n_class = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]
        y = np.tile(np.arange(n_class)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, n_class), tf.float32)

        target_inds = tf.reshape(tf.range(n_class), [n_class, 1])
        target_inds = tf.tile(target_inds, [1, n_query])

        cat = tf.concat([
            tf.reshape(support, [n_class * n_support, self.w, self.h, self.c]),
            tf.reshape(query, [n_class * n_query, self.w, self.h, self.c])], axis=0)
        z = self.encoder(cat)

        z_prototypes = tf.reshape(z[:n_class * n_support], [n_class, n_support, z.shape[-1]])
        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
        z_query = z[n_class * n_support:]

        dists = calc_euclidian_dists(z_query, z_prototypes)

        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [n_class, n_query, -1])

        loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        eq = tf.cast(tf.equal(tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32), tf.cast(y, tf.int32)), tf.float32)
        acc = tf.reduce_mean(eq)

        return loss, acc

    def save(self, model_path):
        """
        Save encoder to the file.

        Args:
            model_path (str): path to the .h5 file.

        Returns: None

        """
        self.encoder.save(model_path)

    def load(self, model_path):
        """
        Load encoder from the file.

        Args:
            model_path (str): path to the .h5 file.

        Returns: None

        """
        self.encoder(tf.zeros([1, self.w, self.h, self.c]))
        self.encoder.load_weights(model_path)