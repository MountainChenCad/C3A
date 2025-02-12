from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, TimeDistributed
from tqdm import tqdm

from utils.tensor_operations import *
from utils.distance_functions import *


class ProtoShotXAI:
    def __init__(self, model, input_layer=0, feature_layer=-2, class_layer=-1):
        if class_layer is not None:
            self.class_weights = model.layers[class_layer].get_weights()[0]
            self.class_bias = model.layers[class_layer].get_weights()[1]
        else:
            self.class_weights = None
            self.class_bias = None

        input_shape = model.input.shape

        output_vals = model.layers[feature_layer].output
        model = Model(inputs=model.input, outputs=output_vals)

        model_5d = TimeDistributed(model)

        support = Input(input_shape)
        support_features = model_5d(support)
        support_features = Lambda(reduce_tensor)(support_features)

        query = Input(input_shape)
        query_features = model_5d(query)
        query_features = Lambda(reshape_query)(query_features)

        features = Lambda(cosine_dist_features)([support_features, query_features])  # negative distance
        self.model = Model([support, query], features)

    def compute_score_from_features(self, features, iclass):
        s_feature_t, q_feature_t, s_feature_norm, q_feature_norm = features
        s_feature_t = s_feature_t.numpy()
        q_feature_t = q_feature_t.numpy()
        # We have change this to .any() for the script to run, avoiding error.
        # if self.class_weights.any()!=0:
        # # if self.class_weights != None:
        #     s_feature_t = s_feature_t*np.tile(np.expand_dims(self.class_weights[:,iclass],axis=(0,1)),(s_feature_t.shape[0],s_feature_t.shape[1],1))
        #     q_feature_t = q_feature_t*np.tile(np.expand_dims(self.class_weights[:,iclass],axis=(0,1)),(q_feature_t.shape[0],q_feature_t.shape[1],1))

        s_feature_norm = np.sqrt(np.sum(s_feature_t * s_feature_t, axis=-1))
        q_feature_norm = np.sqrt(np.sum(q_feature_t * q_feature_t, axis=-1))
        den = s_feature_norm * q_feature_norm
        a = np.sum(s_feature_t * q_feature_t, axis=-1) / den
        score = np.squeeze(np.sum(s_feature_t * q_feature_t, axis=-1) / den)

        return score

    def compute_score_from_features_dn4(self, features, iclass, k=1):
        """
        Compute the image-to-class similarity score for DN4.

        Args:
            features: Tuple containing support and query features, and their norms.
                      (s_feature_t, q_feature_t, s_feature_norm, q_feature_norm)
            iclass: The class index for which the similarity is being computed.
            k: Number of nearest neighbors to consider (default is 1).

        Returns:
            score: The image-to-class similarity score.
        """
        s_feature_t, q_feature_t, s_feature_norm, q_feature_norm = features

        # Convert tensors to numpy arrays for computation
        s_feature_t = s_feature_t.numpy()
        q_feature_t = q_feature_t.numpy()

        # Normalize the support and query features
        s_feature_norm = np.sqrt(np.sum(s_feature_t * s_feature_t, axis=-1, keepdims=True))
        q_feature_norm = np.sqrt(np.sum(q_feature_t * q_feature_t, axis=-1, keepdims=True))
        s_feature_t = s_feature_t / (s_feature_norm + 1e-8)
        q_feature_t = q_feature_t / (q_feature_norm + 1e-8)

        # Flatten the support features into a pool of local descriptors
        s_feature_pool = s_feature_t.reshape(-1, s_feature_t.shape[-1])  # Shape: (num_support_descriptors, d)

        # Initialize the score
        score = 0.0

        # Compute the similarity for each local descriptor in the query image
        for q_descriptor in q_feature_t.reshape(-1, q_feature_t.shape[-1]):  # Shape: (num_query_descriptors, d)
            # Compute cosine similarity between the query descriptor and all support descriptors
            similarities = np.dot(s_feature_pool, q_descriptor)  # Shape: (num_support_descriptors,)

            # Find the top-k nearest neighbors
            top_k_indices = np.argsort(similarities)[-k:]  # Indices of the k largest similarities
            top_k_similarities = similarities[top_k_indices]  # Retrieve the top-k similarities

            # Aggregate the top-k similarities (e.g., sum or mean)
            score += np.sum(top_k_similarities)  # Sum of top-k similarities

        # Normalize the score by the number of query descriptors
        num_query_descriptors = q_feature_t.shape[0] * q_feature_t.shape[1]
        score /= num_query_descriptors

        return score

    def compute_features(self, support_data_expand, query_expand, iclass):

        features = self.model([support_data_expand, query_expand])
        s_feature_t, q_feature_t, s_feature_norm, q_feature_norm = features
        s_feature_t = s_feature_t.numpy()
        q_feature_t = q_feature_t.numpy()
        if self.class_weights != None:
            s_feature_t = s_feature_t * np.tile(np.expand_dims(self.class_weights[:, iclass], axis=(0, 1)),
                                                (s_feature_t.shape[0], s_feature_t.shape[1], 1))
            q_feature_t = q_feature_t * np.tile(np.expand_dims(self.class_weights[:, iclass], axis=(0, 1)),
                                                (q_feature_t.shape[0], q_feature_t.shape[1], 1))

        s_feature_norm = np.sqrt(np.sum(s_feature_t * s_feature_t, axis=-1))
        q_feature_norm = np.sqrt(np.sum(q_feature_t * q_feature_t, axis=-1))
        den = s_feature_norm * q_feature_norm

        return s_feature_t, q_feature_t, den

    def compute_score(self, support_data_expand, query_expand, class_indx):
        # query_expand = np.expand_dims(np.copy(query),axis=0) # Batch size of 1
        # support_data_expand = np.expand_dims(np.copy(support_data),axis=0) # Only 1 support set

        features = self.model([support_data_expand, query_expand])
        scores = self.compute_score_from_features(features, class_indx)
        return scores

    def image_feature_attribution(self, support_data, query, class_indx, ref_pixel, pad=4, progress_bar=True):
        rows = np.shape(query)[1]
        cols = np.shape(query)[2]
        chnls = np.shape(query)[3]

        # Added batch diemnsion (no actual change)
        query_expand = np.expand_dims(np.copy(query), axis=0)  # Batch size of 1
        support_data_expand = np.expand_dims(np.copy(support_data), axis=0)  # Only 1 support set

        features = self.model([support_data_expand, query_expand])
        # ref_score = self.compute_score_from_features_dn4(features,class_indx)
        ref_score = self.compute_score_from_features(features, class_indx)
        print(ref_score, class_indx)
        # Create peturbed_images
        score_matrix = np.zeros((rows, cols))
        peturbed_images = np.zeros((cols, rows, cols, chnls))
        for ii in tqdm(range(rows), disable=(not progress_bar)):
            for jj in range(cols):
                peturbed_images[jj, :, :, :] = np.copy(query)
                min_ii = np.max([ii - pad, 0])
                max_ii = np.min([ii + pad, rows])
                min_jj = np.max([jj - pad, 0])
                max_jj = np.min([jj + pad, cols])
                for ichnl in range(chnls):
                    peturbed_images[jj, min_ii:max_ii, min_jj:max_jj, ichnl] = ref_pixel[ichnl]

            peturbed_images_expand = np.expand_dims(np.copy(peturbed_images), axis=0)
            features = self.model([support_data_expand, peturbed_images_expand])

            scores = self.compute_score_from_features(features, class_indx)
            score_matrix[ii, :] = ref_score - scores

        return score_matrix

    def image_feature_attribution_dn4(self, support_data, query, class_indx, ref_pixel, pad=4, progress_bar=True):
        rows = np.shape(query)[1]
        cols = np.shape(query)[2]
        chnls = np.shape(query)[3]

        # Added batch diemnsion (no actual change)
        query_expand = np.expand_dims(np.copy(query), axis=0)  # Batch size of 1
        for i in range(support_data.shape[0]):
            support_data_expand = np.expand_dims(np.copy(support_data[i]), axis=0)  # Only 1 support set
            support_data_expand = np.expand_dims(np.copy(support_data_expand[i]), axis=0)
            features = self.model([support_data_expand, query_expand])

        # ref_score = self.compute_score_from_features_dn4(features,class_indx)
        ref_score = self.compute_score_from_features(features, class_indx)

        # Create peturbed_images
        score_matrix = np.zeros((rows, cols))
        peturbed_images = np.zeros((cols, rows, cols, chnls))
        for ii in tqdm(range(rows), disable=(not progress_bar)):
            for jj in range(cols):
                peturbed_images[jj, :, :, :] = np.copy(query)
                min_ii = np.max([ii - pad, 0])
                max_ii = np.min([ii + pad, rows])
                min_jj = np.max([jj - pad, 0])
                max_jj = np.min([jj + pad, cols])
                for ichnl in range(chnls):
                    peturbed_images[jj, min_ii:max_ii, min_jj:max_jj, ichnl] = ref_pixel[ichnl]

            peturbed_images_expand = np.expand_dims(np.copy(peturbed_images), axis=0)
            features = self.model([support_data_expand, peturbed_images_expand])
            # scores = self.compute_score_from_features_dn4(features,class_indx)
            scores = self.compute_score_from_features(features, class_indx)
            score_matrix[ii, :] = ref_score - scores

        return score_matrix


