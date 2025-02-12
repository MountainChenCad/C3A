# Source: https://keras.io/examples/vision/grad_cam/
import numpy as np
from heapq import nlargest
from os import path
from tensorflow.keras.models import  Model
import tensorflow as tf
from protonet_tf2.protonet.models.prototypical import calc_euclidian_dists

class GradCam:
    def __init__(self, model, last_conv_layer_name, support, n_class=5):
        self.n_class = support.shape[0]
        self.n_support = support.shape[1]
        self.support = model(tf.reshape(support, [self.n_class * self.n_support, 84, 84, 3]))
        self.grad_model = Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
    def make_gradcam_heatmap(self, query, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        n_query = query.shape[1]
        img_array = tf.reshape(query, [1 * n_query, 84, 84, 3])

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, query = self.grad_model(img_array)
            z = tf.concat([self.support, query], axis=0)
            z_prototypes = tf.reshape(z[:self.n_class * self.n_support],
                        [self.n_class, self.n_support, z.shape[-1]])
            z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
            z_query = z[self.n_class * self.n_support:]
            preds = -calc_euclidian_dists(z_query, z_prototypes)
            # preds = tf.nn.log_softmax(-dists, axis=-1)
            # preds = tf.exp(log_p_y)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()