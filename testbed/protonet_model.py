import tensorflow as tf
from tensorflow.keras import Model
from protonet_tf2.protonet.models.prototypical import calc_euclidian_dists
class Proto(Model):
    def __init__(self, encoder, support, **kwargs):
        super(Proto, self).__init__(**kwargs)
        self.encoder = encoder
        self.support = support

    def call(self, query):

        n_class = self.support.shape[0]
        n_support = self.support.shape[1]
        n_query = query.shape[1]

        # 将 support 和 query 合并并通过编码器
        cat = tf.concat([
            tf.reshape(self.support, [n_class * n_support, 84, 84, 3]),
            tf.reshape(query, [1 * query.shape[1], 84, 84, 3])
        ], axis=0)
        z = self.encoder(cat)

        # 计算 prototypes
        z_prototypes = tf.reshape(z[:n_class * n_support], [n_class, n_support, z.shape[-1]])
        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)

        # 提取 query 的编码
        z_query = z[n_class * n_support:]

        # 计算欧氏距离
        dists = calc_euclidian_dists(z_query, z_prototypes)

        # 计算 log softmax
        log_p_y = tf.nn.log_softmax(-dists, axis=-1)

        # 返回预测结果
        return tf.exp(log_p_y)

if __name__ == '__main__':

    # 假设有一个编码器模型
    encoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(84, 84, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128)
    ])

    # 假设输入数据
    support_data = tf.random.normal((5, 1, 84, 84, 3))  # 10类，每类5个支持样本
    query_data = tf.random.normal((1, 1, 84, 84, 3))   # 1个查询集，20个查询样本
    # 初始化 Proto 模型
    proto_model = Proto(encoder, support_data)
    # 获取预测结果
    predictions = proto_model(query_data)
    print(predictions.shape)  # 输出: (20, 10)