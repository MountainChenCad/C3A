# ------------------------------------------------------------------------------
# Author: Lingfeng Chen
# Date: 2025.02.11
# Description: Tools for fidelity calculation.
# ------------------------------------------------------------------------------

import numpy as np
from skimage.transform import resize as imresize
from skimage.filters import gaussian
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed
from utils.distance_functions import *
from protonet_tf2.protonet.models.prototypical import calc_euclidian_dists
def generate_masked_images(feature_attributions, original_image, sigma=10, input_percentile=99.9, mask_threshold=0.5):
    """
    Generate masked images for Fidelity+ and Fidelity- using Gaussian-blurred background.

    Args:
        feature_attributions (np.array): The importance map.
        original_image (np.array): The original image.
        sigma (float): Standard deviation for Gaussian blur.
        input_percentile (float): Percentile for normalization.
        mask_threshold (float): Threshold for binarizing the importance map.

    Returns:
        masked_plus (np.array): Masked image for Fidelity+.
        masked_minus (np.array): Masked image for Fidelity-.
    """
    # Preprocess feature attributions
    if len(feature_attributions.shape) > 2:
        feature_attributions = np.mean(feature_attributions, axis=-1)
        feature_attributions = np.squeeze(feature_attributions)

        # Resize feature attributions to match the original image
    original_shape = original_image.shape[:2]
    if feature_attributions.shape != original_shape:
        feature_attributions = imresize(feature_attributions, original_shape, anti_aliasing=True)

        # Normalize feature attributions
    abs_vals = np.abs(feature_attributions)
    max_val = np.nanpercentile(abs_vals, input_percentile)

    # Calculate vmin and vmax for normalization
    vmin = -max_val
    vmax = max_val

    # Normalize the feature attributions
    normalized = (feature_attributions - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)  # Ensure values are between 0 and 1

    # Generate binary mask
    mask = (normalized >= mask_threshold).astype(np.float32)

    # Expand dimensions to match image channels
    mask_plus = (1 - mask)[..., np.newaxis]  # Fidelity+ uses complementary mask
    mask_minus = mask[..., np.newaxis]  # Fidelity- uses original mask

    # Generate Gaussian-blurred background
    blurred_background = gaussian(original_image, sigma=sigma, multichannel=True, preserve_range=True)

    # Generate masked images
    masked_plus = original_image * mask_plus + blurred_background * (1 - mask_plus)
    masked_minus = original_image * mask_minus + blurred_background * (1 - mask_minus)

    return masked_plus, masked_minus

class Embed:
    def __init__(self, model, feature_layer=-4):
        self.encoder = model
        input_shape = model.input.shape
        output_vals = model.layers[feature_layer].output
        model = Model(inputs=model.input, outputs=output_vals)

        model_5d = TimeDistributed(model)

        support = Input(input_shape)
        support_features = model_5d(support)

        query = Input(input_shape)
        query_features = model_5d(query)

        features = ([support_features, query_features])  # negative distance
        self.model = Model([support, query], features)

    def compute_similarity_score(self, support, query, idx):
        support = tf.cast(support, tf.float64)
        query = tf.cast(query, tf.float64)
        n_class = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]

        cat = tf.concat([
            tf.reshape(support, [n_class * n_support, 84, 84, 3]),
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
        # score = tf.exp(log_p_y)[0][idx]
        score = log_p_y[0][idx]
        return score

    def fidelity_scores(self, support_data, query, query_plus, query_minus, idx=0):

        # Added batch diemnsion (no actual change)
        query_expand = np.expand_dims(np.copy(query), axis=0)  # Batch size of 1
        query_plus_expand = np.expand_dims(np.copy(query_plus), axis=0)  # Batch size of 1
        query_minus_expand = np.expand_dims(np.copy(query_minus), axis=0)  # Batch size of 1

        if np.ndim(support_data) == 5:
            support_data_expand = support_data
        else:
            support_data_expand = np.expand_dims(np.copy(support_data), axis=0)  # Only 1 support set

        # features_ref = self.model([support_data_expand, query_expand])
        # features_plus = self.model([support_data_expand, query_plus_expand])
        # features_minus = self.model([support_data_expand, query_minus_expand])
        ref_score = self.compute_similarity_score(support_data_expand, query_expand, idx)
        plus_score = ref_score - self.compute_similarity_score(support_data_expand, query_plus_expand, idx)
        minus_score = ref_score - self.compute_similarity_score(support_data_expand, query_minus_expand, idx)

        return plus_score, minus_score


# if __name__ == '__main__':

    ### Ploting functions.
    # plt = xai_plot(c3a_target1_scores, resize_the_batch(query_pickle)[0])
    # plt.savefig(
    #     f"./results/{dataset_str}_feature_attribution_map/c3a_{target1_name}_Features_{input_model_str}_{shot}shot.png",
    #     dpi=450)
    #
    # plt = xai_plot(c3a_target2_scores, resize_the_batch(query_pickle)[0])
    # plt.savefig(
    #     f"./results/{dataset_str}_feature_attribution_map/c3a_{target2_name}_Features_{input_model_str}_{shot}shot.png",
    #     dpi=450)