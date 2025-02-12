from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, TimeDistributed
from tqdm import tqdm

from utils.tensor_operations import *
from utils.distance_functions import *

class ProtoShotXAI:
    def __init__(self, model, input_layer=0, feature_layer=-4, class_layer=-1):

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


    def compute_score_from_features(self,features,iclass):
        ## This methods is ProtoShotXAI, which squeezes spatial dimensions
        s_feature_t, q_feature_t = features
        s_feature_t = s_feature_t.numpy()  # 形状: (1, 100, 7, 7, 2048)
        q_feature_t = q_feature_t.numpy()  # 形状: (1, 1, 7, 7, 2048)

        s_feature_pooled = np.mean(s_feature_t, axis=(2, 3))  # 形状: (1, 100, 2048)
        q_feature_pooled = np.mean(q_feature_t, axis=(2, 3))  # 形状: (1, 1, 2048)
        # s_feature_pooled = s_feature_t  # 形状: (1, 100, 2048)
        # q_feature_pooled = q_feature_t  # 形状: (1, 1, 2048)

        s_feature_avg = np.mean(s_feature_pooled, axis=1)  # 形状: (1, 2048)

        s_feature_norm = np.linalg.norm(s_feature_avg, axis=-1, keepdims=True)  # 形状: (1, 1)
        q_feature_norm = np.linalg.norm(q_feature_pooled, axis=-1, keepdims=True)  # 形状: (1, 1)
        dot_product = np.sum(s_feature_avg * q_feature_pooled, axis=-1, keepdims=True)  # 形状: (1, 1)
        similarity = dot_product / (s_feature_norm * q_feature_norm)  # 形状: (1, 1)

        score = np.squeeze(similarity)  # 形状: (1,)
        return score

    def compute_score_from_features_localshot(self, features, iclass, k=10):
        ## This method implements DN4, which uses nearest neighbor similarity
        s_feature_t, q_feature_t = features
        s_feature_t = s_feature_t.numpy()  # 形状: (1, batchsize_s, 7, 7, 2048)
        q_feature_t = q_feature_t.numpy()  # 形状: (1, batchsize_q, 7, 7, 2048)

        s_feature_t = np.squeeze(s_feature_t, axis=0)  # 形状: (batchsize_s, 7, 7, 2048)
        q_feature_t = np.squeeze(q_feature_t, axis=0)  # 形状: (batchsize_q, 7, 7, 2048)

        batchsize_q = q_feature_t.shape[0]
        batchsize_s = s_feature_t.shape[0]
        size_q_flat_last = q_feature_t.shape[-1]
        size_s_flat_last = s_feature_t.shape[-1]

        s_feature_flat = s_feature_t.reshape(-1, size_s_flat_last)  # 形状: (batchsize_s * 7 * 7, 2048)
        q_feature_flat = q_feature_t.reshape(batchsize_q, -1, size_q_flat_last)  # 形状: (batchsize_q, 7 * 7, 2048)

        s_feature_norm = np.linalg.norm(s_feature_flat, axis=-1, keepdims=True)  # 形状: (batchsize_s * 7 * 7, 1)
        q_feature_norm = np.linalg.norm(q_feature_flat, axis=-1, keepdims=True)  # 形状: (batchsize_q, 7 * 7, 1)
        s_feature_flat = s_feature_flat / (s_feature_norm + 1e-8)  # 形状: (batchsize_s * 7 * 7, 2048)
        q_feature_flat = q_feature_flat / (q_feature_norm + 1e-8)  # 形状: (batchsize_q, 7 * 7, 2048)

        similarity = np.matmul(q_feature_flat, s_feature_flat.T)  # 形状: (batchsize_q, 7 * 7, batchsize_s * 7 * 7)

        top_k_similarity = np.sort(similarity, axis=-1)[:, :, -k:]  # 形状: (batchsize_q, 7 * 7, k)

        score = np.sum(top_k_similarity, axis=(-1, -2))  # 形状: (batchsize_q,)

        return score

    def image_feature_attribution(self,support_data,query, class_indx, ref_pixel, pad=2 , progress_bar=True):
        rows = np.shape(query)[1]
        cols = np.shape(query)[2]
        chnls = np.shape(query)[3]

        # Added batch diemnsion (no actual change)
        query_expand = np.expand_dims(np.copy(query),axis=0) # Batch size of 1
        support_data_expand = np.expand_dims(np.copy(support_data),axis=0) # Only 1 support set

        features = self.model([support_data_expand,query_expand])
        # ref_score = self.compute_score_from_features_dn4(features,class_indx)
        ref_score = self.compute_score_from_features(features, class_indx)
        print(ref_score, class_indx)
        # Create peturbed_images
        score_matrix = np.zeros((rows,cols))
        peturbed_images = np.zeros((cols,rows,cols,chnls))
        for ii in tqdm(range(rows),disable=(not progress_bar)):
            for jj in range(cols):
                peturbed_images[jj,:,:,:] = np.copy(query)
                min_ii = np.max([ii-pad,0])
                max_ii = np.min([ii+pad,rows])
                min_jj = np.max([jj-pad,0])
                max_jj = np.min([jj+pad,cols])
                for ichnl in range(chnls):
                    peturbed_images[jj,min_ii:max_ii,min_jj:max_jj,ichnl] = ref_pixel[ichnl]
            
            peturbed_images_expand = np.expand_dims(np.copy(peturbed_images),axis=0)
            features = self.model([support_data_expand,peturbed_images_expand])

            scores = self.compute_score_from_features(features, class_indx)
            score_matrix[ii,:] = ref_score - scores
        
        return score_matrix

    def image_feature_attribution_localshot(self, support_data, query, class_indx, ref_pixel, pad=2, progress_bar=True):
        rows = np.shape(query)[1]
        cols = np.shape(query)[2]
        chnls = np.shape(query)[3]

        # Added batch diemnsion (no actual change)
        query_expand = np.expand_dims(np.copy(query), axis=0)  # Batch size of 1
        support_data_expand = np.expand_dims(np.copy(support_data), axis=0)  # Only 1 support set

        features = self.model([support_data_expand, query_expand])
        # ref_score = self.compute_score_from_features_dn4(features,class_indx)
        ref_score = self.compute_score_from_features_localshot(features, class_indx)
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

            scores = self.compute_score_from_features_localshot(features, class_indx)
            score_matrix[ii, :] = ref_score - scores

        return score_matrix

    def image_feature_attribution_contra(self, support_data_1, support_data_2,
                                         support_data_3,
                                         query, class_indx, ref_pixel, pad=2, alpha=0.5, progress_bar=True):
        rows = np.shape(query)[1]
        cols = np.shape(query)[2]
        chnls = np.shape(query)[3]

        # Added batch diemnsion (no actual change)
        query_expand = np.expand_dims(np.copy(query), axis=0)  # Batch size of 1
        support_data_1_expand = np.expand_dims(np.copy(support_data_1), axis=0)  # Only 1 support set
        support_data_2_expand = np.expand_dims(np.copy(support_data_2), axis=0)  # Only 1 support set
        # support_data_3_expand = np.expand_dims(np.copy(support_data_3), axis=0)

        features_1 = self.model([support_data_1_expand, query_expand])
        features_2 = self.model([support_data_2_expand, query_expand])
        # features_3 = self.model([support_data_3_expand, query_expand])

        # ref_score = (self.compute_score_from_features_localshot(features_1, class_indx)
        #              - self.compute_score_from_features_localshot(features_2, class_indx)
        #              - 0.1*self.compute_score_from_features_localshot(features_3, class_indx))
        ref_score = ((1 - alpha) * self.compute_score_from_features_localshot(features_1, class_indx)
                     - alpha * self.compute_score_from_features_localshot(features_2, class_indx))
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
            features_1 = self.model([support_data_1_expand, peturbed_images_expand])
            features_2 = self.model([support_data_2_expand, peturbed_images_expand])
            # features_3 = self.model([support_data_3_expand, peturbed_images_expand])

            scores = ((1 - alpha) * self.compute_score_from_features_localshot(features_1, class_indx)
                      - alpha * self.compute_score_from_features_localshot(features_2, class_indx))
            # scores = (self.compute_score_from_features_localshot(features_1, class_indx)
            #           - self.compute_score_from_features_localshot(features_2, class_indx)
            #           - 0.1*self.compute_score_from_features_localshot(features_3, class_indx))
            score_matrix[ii, :] = ref_score - scores

        return score_matrix

