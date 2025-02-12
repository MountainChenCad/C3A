import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from protonet_tf2.protonet.models.prototypical import calc_euclidian_dists


class ProtoNetSHAPExplainer:
    def __init__(self, model, last_conv_layer_name, support, shot, n_class=5):
        self.shot = shot
        self.n_class = n_class
        self.n_support = support.shape[1]

        # 创建特征提取模型
        self.feature_model = Model(
            inputs=model.inputs,
            outputs=model.get_layer(last_conv_layer_name).output
        )

        # 预计算支持集特征
        self.prototypes = self._create_prototypes(support)

        # 构建SHAP解释模型
        self.explainer_model = self._build_shap_model()
        self.explainer = shap.GradientExplainer(
            model=self.explainer_model,
            data=self._get_background_data(support),
            local_smoothing=0  # std dev of smoothing noise
        )

    def _create_prototypes(self, support):
        """预计算原型"""
        support_features = self.feature_model.predict(
            tf.reshape(support, [self.n_class * self.n_support, 84, 84, 3])
        )
        return tf.reduce_mean(
            tf.reshape(support_features, [self.n_class, self.n_support, -1]),
            axis=1
        )

    def _build_shap_model(self):
        """构建符合SHAP要求的Keras模型"""
        input_layer = Input(shape=(1, 84, 84, 3))

        # 特征提取
        features = self.feature_model(input_layer[0])
        x_flattened = tf.reshape(features, [1, -1])
        dists = tf.norm(x_flattened - self.prototypes, axis=1)

        return Model(inputs=input_layer, outputs=-dists)

    def _get_background_data(self, support):
        """生成背景数据"""
        return tf.reshape(support, [self.n_class * self.n_support, 84, 84, 3])

    def explain(self, query, class_index=0, visualize=True):
        """生成解释"""
        # 生成SHAP值
        shap_values = self.explainer.shap_values(
            query,
            nsamples = self.n_class * self.shot,
            ranked_outputs=class_index,
        )

        return shap_values


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    # 创建一个简单的卷积神经网络模型
    def create_model():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(84, 84, 3), name='conv1'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', name='conv2'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', name='conv3'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu', name='conv4'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(5, activation='softmax')
        ])
        return model

    # 初始化参数
    n_class = 5
    n_support = 5

    # 生成随机支持集数据
    support_set = np.random.rand(n_class, n_support, 84, 84, 3)

    # 创建模型
    original_model = create_model()

    # 创建解释器
    explainer = ProtoNetSHAPExplainer(
        model=original_model,
        last_conv_layer_name="conv4",
        support=support_set,
        n_class=n_class
    )

    # 生成随机查询图像
    query_image = np.random.rand(1, 84, 84, 3)

    # 生成解释
    shap_values = explainer.explain(query_image, class_index=0, visualize=True)