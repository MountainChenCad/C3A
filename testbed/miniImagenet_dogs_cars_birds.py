# ------------------------------------------------------------------------------
# Author: Lingfeng Chen
# Date: 2025.02.10
# Description: A testbed for miniImagenet and other fine-grained benchmarks.
# ------------------------------------------------------------------------------

import os
import argparse
import pickle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from architectures.protoshotxai import ProtoShotXAI  # ProtoShotXAI
from utils.ploting_function import xai_plot
import matplotlib.pyplot as plt
from protonet_tf2.protonet.models.prototypical import Prototypical, calc_euclidian_dists
from architectures.c3a import C3Amodel  # C3A XAI
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from fidelity_cal import generate_masked_images, Embed
from lime import lime_image  # LIME XAI
from architectures.gradcam import GradCam  # GradCAM++ XAI
from architectures.shap_modified import ProtoNetSHAPExplainer  # SHAP XAI
from SINEX.code.sinex import Sinex
from SINEX.code.sinexc import Sinexc  # SINEX/SINEXC XAI

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def model_wrapper(images):
    """
    Wrapper function to handle a batch of images and pass them to the model for prediction.
    Assumes images input shape is (batch_size, 84, 84, 3).
    Returns predictions with shape (batch_size, class_number).
    """
    # Expand dimensions to match model input requirements
    query = tf.cast(np.expand_dims(images, axis=0), tf.float32)
    support = tf.cast(episode_support_data, tf.float32)

    n_class = support.shape[0]
    n_support = support.shape[1]
    n_query = query.shape[1]
    y = np.tile(np.arange(n_class)[:, np.newaxis], (1, n_query))
    y_onehot = tf.cast(tf.one_hot(y, n_class), tf.float32)

    target_inds = tf.reshape(tf.range(n_class), [n_class, 1])
    target_inds = tf.tile(target_inds, [1, n_query])

    cat = tf.concat([
        tf.reshape(support, [n_class * n_support, 84, 84, 3]),
        tf.reshape(query, [1 * n_query, 84, 84, 3])], axis=0)
    z = base_model.encoder(cat)

    z_prototypes = tf.reshape(z[:n_class * n_support], [n_class, n_support, z.shape[-1]])
    z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
    z_query = z[n_class * n_support:]
    dists = calc_euclidian_dists(z_query, z_prototypes)
    log_p_y = tf.nn.log_softmax(-dists, axis=-1)
    # Collect predictions
    predictions = tf.exp(log_p_y)

    return predictions


def resize_the_batch(data):
    """  
    resize a batch of images to the specified dimensions.

    Parameters:  
    data (numpy.ndarray): A 4D numpy array of shape (batch_size, height, width, channels),  
                         where batch_size is the number of images, height and width   
                         are the dimensions of each image, and channels are the color channels.  

    Returns:  
    numpy.ndarray: A 4D numpy array containing the resized images,
                   with shape (batch_size, 84, 84, channels).  
    """
    batch_size, height, width, channels = data.shape
    resized_data = []

    for i in range(batch_size):
        img_array = data[i]
        img_resized = cv2.resize(img_array, (84, 84), interpolation=cv2.INTER_AREA)
        resized_data.append(img_resized)

    return np.array(resized_data)


def print_ndarray_info(ndarray):
    print(f"Data type: {ndarray.dtype}")
    print(f"Shape: {ndarray.shape}")
    print(f"Number of dimensions: {ndarray.ndim}")
    print(f"Total number of elements: {ndarray.size}")
    print(f"Bytes per element: {ndarray.itemsize}")


def print_dict_info(dictionary):
    # Print keys
    print("Keys:", dictionary.keys())
    # Print values
    print("Values:", dictionary.values())
    # Print key-value pairs
    print("Key-Value Pairs:", dictionary.items())
    # Print dictionary length (number of key-value pairs)
    print("Length:", len(dictionary))
    # Print the full dictionary
    print("Full Dictionary:", dictionary)


if __name__ == '__main__':

    ### This few-shot XAI framwork need you to specify shot number.
    shot = 5
    k = 10
    dataset_str = 'dogs'
    padding_size = 6
    ### In our experiments, we only focus on Conv64F and ResNet12 backbone.
    input_model_str = 'Conv64F'
    # input_model_str = 'ResNet12'
    model_map = {
        'Conv64F': {
            'model_filename': f'../supplementry_materials/weights/conv4_{dataset_str}.h5',
            'feature_layer': -5,
            'flatten_layer': -2,
            'encoder_type': 'conv64F',
            'layer_num': 12,
            'last_conv_name': 'conv2d_3'
        },
        'ResNet12': {
            'model_filename': f'../supplementry_materials/weights/resnet12_{dataset_str}.h5',
            # Load model if needed
            'feature_layer': -4,
            'encoder_type': 'resnet12'
        }
    }
    if input_model_str in model_map:
        model_info = model_map[input_model_str]
        model_filename = model_info['model_filename']
        feature_layer = model_info['feature_layer']
        encoder_type = model_info['encoder_type']
        layer_num = model_info['layer_num']
        last_conv_name = model_info['last_conv_name']
        flatten_layer = model_info['flatten_layer']
    else:
        raise ValueError(f"Model {input_model_str} is not recognized.")

    ### Note that miniImagenet, dogs, cars, and birds are applicable to this testbed.
    save_dir = f"./results/{dataset_str}_feature_attribution_map"
    os.makedirs(save_dir, exist_ok=True)
    dataset_map = {
        'miniImagenet': {
            'query_filename': '../data/french_bulldog_and_tank.pickle',
            'support_filename': '../data/miniImageNet_train_data.npy',
            'target1_name': 'french_bulldog',
            'target2_name': 'tank',
            'target3_name': 'house_finch',
            'target4_name': 'jellyfish',
            'target5_name': 'miniature_poodle',
            'target1_label': 'n02108915',
            'target2_label': 'n04389033',
            'target3_label': 'n01532829',
            'target4_label': 'n01910747',
            'target5_label': 'n02113712',
        },
        'dogs': {
            'query_filename': '../data/chow_and_shih_tzu.pkl',
            'support_filename': '../data/dogs_train_data.npy',
            'target1_name': 'chow',
            'target2_name': 'shih_tzu',
            'target3_name': 'boston_bull',
            'target4_name': 'appenzeller',
            'target5_name': 'dandie_dinmont',
            'target1_label': 'n02112137-chow',
            'target2_label': 'n02086240-Shih-Tzu',
            'target3_label': 'n02096585-Boston_bull',
            'target4_label': 'n02107908-Appenzeller',
            'target5_label': 'n02096437-Dandie_Dinmont',
        },
        'cars': {
            'query_filename': '../data/rolls_and_buick.pkl',
            'support_filename': '../data/cars_train_data.npy',
            'target1_name': 'rolls',
            'target2_name': 'buick',
            'target3_name': 'ford',
            'target4_name': 'chevrolet',
            'target5_name': 'audi',
            'target1_label': '177',
            'target2_label': '48',
            'target3_label': '115',
            'target4_label': '68',
            'target5_label': '23',
        },
        'birds': {
            'query_filename': '../data/pied_kingfisher_and_savannah_sparrow.pkl',
            'support_filename': '../data/birds_train_data.npy',
            'target1_name': 'pied_kingfisher',
            'target2_name': 'savannah_sparrow',
            'target3_name': 'tropical_kingbird',
            'target4_name': 'white_breasted_nuthatch',
            'target5_name': 'pine_grosbeak',
            'target1_label': '081.Pied_Kingfisher',
            'target2_label': '127.Savannah_Sparrow',
            'target3_label': '077.Tropical_Kingbird',
            'target4_label': '094.White_breasted_Nuthatch',
            'target5_label': '056.Pine_Grosbeak',
        }
    }
    if dataset_str in dataset_map:
        dataset_info = dataset_map[dataset_str]
        query_filename = dataset_info['query_filename']
        support_filename = dataset_info['support_filename']
        target1_name = dataset_info['target1_name']
        target2_name = dataset_info['target2_name']
        target1_label = dataset_info['target1_label']
        target2_label = dataset_info['target2_label']
        target3_label = dataset_info['target3_label']
        target4_label = dataset_info['target4_label']
        target5_label = dataset_info['target5_label']
    else:
        raise ValueError(f"Dataset {dataset_str} is not recognized.")

    ### Immediately we load the query and support data here.
    query_dict = pickle.load(open(query_filename, 'rb'))
    # print_dict_info(query_dict)
    query_pickle = np.expand_dims(
        preprocess_input(
            pickle.load(open(query_filename, 'rb'))[f'{target1_name}_and_{target2_name}']),
        axis=0)
    support_dict = np.load(support_filename, allow_pickle=True).item()
    # print_dict_info(support_dict)
    '''
    This dictionary contains two keys: data and labels. The data key holds a4-dimensional NumPy 
    array of shape (38400, 84, 84, 3), representing 38,400 RGB images with a resolution of 84x84 
    pixels each, stored as uint8 values. The labels key contains a 1-dimensional NumPy array of 
    shape (38400, ), where all entries are the string 'n01532829', 'n01532829', 'n01532829', ...,
    'n13133613', indicating that all images belong to the several different categories.
    '''
    support_data, support_labels = support_dict['data'], support_dict['labels']
    support_data_target1, support_data_target2 = (resize_the_batch(
        preprocess_input(support_data[support_labels == target1_label][:shot])),
                                                  resize_the_batch(
                                                      preprocess_input(
                                                          support_data[support_labels == target2_label][:shot])))

    ### Load the associated model weights.
    base_model = Prototypical(w=84, h=84, c=3, nb_layers=4, encoder_type=encoder_type)
    base_model.encoder(tf.keras.Input((84, 84, 3)))
    base_model.encoder.load_weights(model_filename)
    base_model.encoder.summary()
    rgb_query = resize_the_batch(  # For fidelity calculation.
        np.expand_dims(
            pickle.load(open(query_filename, 'rb'))[f'{target1_name}_and_{target2_name}'], axis=0)).squeeze() / 255
    embed = Embed(base_model.encoder, feature_layer=flatten_layer)  # For fidelity calculation.
    query = resize_the_batch(query_pickle)
    ### Create episode data.
    support_data_target3, support_data_target4, support_data_target5 = (
        resize_the_batch(preprocess_input(support_data[support_labels == target3_label][:shot])),
        resize_the_batch(preprocess_input(support_data[support_labels == target4_label][:shot])),
        resize_the_batch(preprocess_input(support_data[support_labels == target5_label][:shot])))
    episode_support_data = np.concatenate(
        [np.expand_dims(support_data_target1, axis=0), np.expand_dims(support_data_target2, axis=0),
         np.expand_dims(support_data_target3, axis=0), np.expand_dims(support_data_target4, axis=0),
         np.expand_dims(support_data_target5, axis=0)], axis=0)
    exclude_support_data = np.concatenate(
        [support_data_target3, support_data_target4,
         support_data_target5], axis=0)
    ref_pixel = query_pickle[0, 0, 0, :]  # Average background pixel after preprocessing.
    # print(f'Average background pixel: {ref_pixel}')
    ### C3A XAI
    c3a_xai = C3Amodel(base_model.encoder, feature_layer=feature_layer, k=k)
    c3a_target1_scores, c3a_target2_scores = (c3a_xai.image_feature_attribution_c3a(
        support_data_1=support_data_target1,
        support_data_2=support_data_target2,
        support_data_3=exclude_support_data,
        query=query, ref_pixel=ref_pixel, pad=padding_size),
                                              c3a_xai.image_feature_attribution_c3a(
                                                  support_data_1=support_data_target2,
                                                  support_data_2=support_data_target1,
                                                  support_data_3=exclude_support_data,
                                                  query=query, ref_pixel=ref_pixel, pad=padding_size))
    ### Ploting functions.
    plt = xai_plot(c3a_target1_scores, resize_the_batch(query_pickle)[0])
    plt.savefig(
        f"./results/{dataset_str}_feature_attribution_map/c3a_{target1_name}_Features_{input_model_str}_{shot}shot_{padding_size}size.png",
        dpi=450)
    plt = xai_plot(c3a_target2_scores, resize_the_batch(query_pickle)[0])
    plt.savefig(
        f"./results/{dataset_str}_feature_attribution_map/c3a_{target2_name}_Features_{input_model_str}_{shot}shot_{padding_size}size.png",
        dpi=450)
    ## Fidelity calculation.
    c3a_target1_plus_image, c3a_target1_minus_image = generate_masked_images(
        feature_attributions=c3a_target1_scores, original_image=rgb_query,
        input_percentile=99.9, mask_threshold=0.5)
    c3a_target2_plus_image, c3a_target2_minus_image = generate_masked_images(
        feature_attributions=c3a_target2_scores, original_image=rgb_query,
        input_percentile=99.9, mask_threshold=0.5)
    c3a_target1_fidelity_plus, c3a_target1_fidelity_minus = embed.fidelity_scores(
        episode_support_data, query,
        resize_the_batch(np.expand_dims(c3a_target1_plus_image, axis=0)),
        resize_the_batch(np.expand_dims(c3a_target1_minus_image, axis=0)), idx=0)
    c3a_target2_fidelity_plus, c3a_target2_fidelity_minus = embed.fidelity_scores(
        episode_support_data, query,
        resize_the_batch(np.expand_dims(c3a_target2_plus_image, axis=0)),
        resize_the_batch(np.expand_dims(c3a_target2_minus_image, axis=0)), idx=1)
    print(f'C3A_{target1_name} -> Fidelity+:{c3a_target1_fidelity_plus}, Fidelity-:{c3a_target1_fidelity_minus}')
    print(f'C3A_{target2_name} -> Fidelity+:{c3a_target2_fidelity_plus}, Fidelity-:{c3a_target2_fidelity_minus}')
    #
    ### ProtoShotXAI
    protoshot_xai = ProtoShotXAI(base_model.encoder, feature_layer=flatten_layer)
    protoshot_target1_scores, protoshot_target2_scores = (protoshot_xai.image_feature_attribution(
        support_data=support_data_target1, query=query, ref_pixel=ref_pixel,
        pad=padding_size
    ),
                                                          protoshot_xai.image_feature_attribution(
                                                              support_data=support_data_target2, query=query,
                                                              ref_pixel=ref_pixel,
                                                              pad=padding_size
                                                          ))
    ### Ploting functions.
    plt = xai_plot(protoshot_target1_scores, resize_the_batch(query_pickle)[0])
    plt.savefig(
        f"./results/{dataset_str}_feature_attribution_map/protoshot_{target1_name}_Features_{input_model_str}_{shot}shot_{padding_size}size.png",
        dpi=450)

    plt = xai_plot(protoshot_target2_scores, resize_the_batch(query_pickle)[0])
    plt.savefig(
        f"./results/{dataset_str}_feature_attribution_map/protoshot_{target2_name}_Features_{input_model_str}_{shot}shot_{padding_size}size.png",
        dpi=450)
    ## Fidelity calculation.
    protoshot_target1_plus_image, protoshot_target1_minus_image = generate_masked_images(
        feature_attributions=protoshot_target1_scores, original_image=rgb_query,
        input_percentile=99.9, mask_threshold=0.5)
    protoshot_target2_plus_image, protoshot_target2_minus_image = generate_masked_images(
        feature_attributions=protoshot_target2_scores, original_image=rgb_query,
        input_percentile=99.9, mask_threshold=0.5)
    protoshot_target1_fidelity_plus, protoshot_target1_fidelity_minus = embed.fidelity_scores(
        episode_support_data, query,
        resize_the_batch(np.expand_dims(protoshot_target1_plus_image, axis=0)),
        resize_the_batch(np.expand_dims(protoshot_target1_minus_image, axis=0)), idx=0)
    protoshot_target2_fidelity_plus, protoshot_target2_fidelity_minus = embed.fidelity_scores(
        episode_support_data, query,
        resize_the_batch(np.expand_dims(protoshot_target2_plus_image, axis=0)),
        resize_the_batch(np.expand_dims(protoshot_target2_minus_image, axis=0)), idx=1)
    print(
        f'ProtoShotXAI_{target1_name} -> Fidelity+:{protoshot_target1_fidelity_plus}, Fidelity-:{protoshot_target1_fidelity_minus}')
    print(
        f'ProtoShotXAI_{target2_name} -> Fidelity+:{protoshot_target2_fidelity_plus}, Fidelity-:{protoshot_target2_fidelity_minus}')

    ## LIME XAI
    explainer = lime_image.LimeImageExplainer()
    lime_explanation = explainer.explain_instance(
        image=query.squeeze(),
        classifier_fn=model_wrapper,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )
    lime_index_target1 = np.where(np.array(lime_explanation.top_labels) == 0)[0][0]
    lime_index_target2 = np.where(np.array(lime_explanation.top_labels) == 1)[0][0]
    ### Attribution calculation and ploting.
    _, lime_target1_attributions = lime_explanation.get_image_and_mask(lime_explanation.top_labels[lime_index_target1],
                                                                       positive_only=False, num_features=10,
                                                                       hide_rest=False)
    plt = xai_plot(lime_target1_attributions, query[0])
    plt.savefig(
        f"./results/{dataset_str}_feature_attribution_map/lime_{target1_name}_Features_{input_model_str}_{shot}shot.png",
        dpi=450)
    _, lime_target2_attributions = lime_explanation.get_image_and_mask(lime_explanation.top_labels[lime_index_target2],
                                                                       positive_only=False, num_features=10,
                                                                       hide_rest=False)
    plt = xai_plot(lime_target2_attributions, query[0])
    plt.savefig(
        f"./results/{dataset_str}_feature_attribution_map/lime_{target2_name}_Features_{input_model_str}_{shot}shot.png",
        dpi=450)
    ### Fidelity calculation.
    lime_target1_plus_image, lime_target1_minus_image = generate_masked_images(
        feature_attributions=lime_target1_attributions, original_image=rgb_query,
        input_percentile=99.9, mask_threshold=0.5)
    lime_target2_plus_image, lime_target2_minus_image = generate_masked_images(
        feature_attributions=lime_target2_attributions, original_image=rgb_query,
        input_percentile=99.9, mask_threshold=0.5)
    lime_target1_fidelity_plus, lime_target1_fidelity_minus = embed.fidelity_scores(
        episode_support_data, query,
        resize_the_batch(np.expand_dims(lime_target1_plus_image, axis=0)),
        resize_the_batch(np.expand_dims(lime_target1_minus_image, axis=0)), idx=0)
    lime_target2_fidelity_plus, lime_target2_fidelity_minus = embed.fidelity_scores(
        episode_support_data, query,
        resize_the_batch(np.expand_dims(lime_target2_plus_image, axis=0)),
        resize_the_batch(np.expand_dims(lime_target2_minus_image, axis=0)), idx=1)
    print(f'LIME_{target1_name} -> Fidelity+:{lime_target1_fidelity_plus}, Fidelity-:{lime_target1_fidelity_minus}')
    print(f'LIME_{target2_name} -> Fidelity+:{lime_target2_fidelity_plus}, Fidelity-:{lime_target2_fidelity_minus}')

    ## SHAP XAI
    explainer = ProtoNetSHAPExplainer(
        model=base_model.encoder,
        last_conv_layer_name=last_conv_name,
        support=episode_support_data,
        shot=shot,
        n_class=5
    )
    shap_target1_attributions = explainer.explain(query, class_index=0)
    plt = xai_plot(shap_target1_attributions, query[0])
    plt.savefig(
        f"./results/{dataset_str}_feature_attribution_map/shap_{target1_name}_Features_{input_model_str}_{shot}shot.png",
        dpi=450)
    shap_target2_attributions = explainer.explain(query, class_index=1)
    plt = xai_plot(shap_target2_attributions, query[0])
    plt.savefig(
        f"./results/{dataset_str}_feature_attribution_map/shap_{target2_name}_Features_{input_model_str}_{shot}shot.png",
        dpi=450)
    ### Fidelity calculation.
    shap_target1_plus_image, shap_target1_minus_image = generate_masked_images(
        feature_attributions=shap_target1_attributions, original_image=rgb_query,
        input_percentile=99.9, mask_threshold=0.5)
    shap_target2_plus_image, shap_target2_minus_image = generate_masked_images(
        feature_attributions=shap_target2_attributions, original_image=rgb_query,
        input_percentile=99.9, mask_threshold=0.5)
    shap_target1_fidelity_plus, shap_target1_fidelity_minus = embed.fidelity_scores(
        episode_support_data, query,
        resize_the_batch(np.expand_dims(shap_target1_plus_image, axis=0)),
        resize_the_batch(np.expand_dims(shap_target1_minus_image, axis=0)), idx=0)
    shap_target2_fidelity_plus, shap_target2_fidelity_minus = embed.fidelity_scores(
        episode_support_data, query,
        resize_the_batch(np.expand_dims(shap_target2_plus_image, axis=0)),
        resize_the_batch(np.expand_dims(shap_target2_minus_image, axis=0)), idx=1)
    print(f'SHAP_{target1_name} -> Fidelity+:{shap_target1_fidelity_plus}, Fidelity-:{shap_target1_fidelity_minus}')
    print(f'SHAP_{target2_name} -> Fidelity+:{shap_target2_fidelity_plus}, Fidelity-:{shap_target2_fidelity_minus}')

    ## GradCAM++ XAI
    # Find last conv laver (Use pre-determined layers instead).
    last_conv_layer_name = base_model.encoder.layers[layer_num].name
    gradcam = GradCam(base_model.encoder, last_conv_layer_name, episode_support_data)
    gradcam_target1_attributions = gradcam.make_gradcam_heatmap(np.expand_dims(query, axis=0), pred_index=0)
    plt = xai_plot(gradcam_target1_attributions, query[0])
    plt.savefig(
        f"./results/{dataset_str}_feature_attribution_map/gardcam_{target1_name}_Features_{input_model_str}_{shot}shot.png",
        dpi=450)
    gradcam_target2_attributions = gradcam.make_gradcam_heatmap(np.expand_dims(query, axis=0), pred_index=1)
    plt = xai_plot(gradcam_target2_attributions, query[0])
    plt.savefig(
        f"./results/{dataset_str}_feature_attribution_map/gardcam_{target2_name}_Features_{input_model_str}_{shot}shot.png",
        dpi=450)
    ### Fidelity calculation.
    gradcam_target1_plus_image, gradcam_target1_minus_image = generate_masked_images(
        feature_attributions=gradcam_target1_attributions, original_image=rgb_query,
        input_percentile=99.9, mask_threshold=0.5)
    gradcam_target2_plus_image, gradcam_target2_minus_image = generate_masked_images(
        feature_attributions=gradcam_target2_attributions, original_image=rgb_query,
        input_percentile=99.9, mask_threshold=0.5)
    gradcam_target1_fidelity_plus, gradcam_target1_fidelity_minus = embed.fidelity_scores(
        episode_support_data, query,
        resize_the_batch(np.expand_dims(gradcam_target1_plus_image, axis=0)),
        resize_the_batch(np.expand_dims(gradcam_target1_minus_image, axis=0)), idx=0)
    gradcam_target2_fidelity_plus, gradcam_target2_fidelity_minus = embed.fidelity_scores(
        episode_support_data, query,
        resize_the_batch(np.expand_dims(gradcam_target2_plus_image, axis=0)),
        resize_the_batch(np.expand_dims(gradcam_target2_minus_image, axis=0)), idx=1)
    print(
        f'GradCAM++_{target1_name} -> Fidelity+:{gradcam_target1_fidelity_plus}, Fidelity-:{gradcam_target1_fidelity_minus}')
    print(
        f'GradCAM++_{target2_name} -> Fidelity+:{gradcam_target2_fidelity_plus}, Fidelity-:{gradcam_target2_fidelity_minus}')

    # ### SINEXC XAI
    # # Set algorithms parameters, input data shape, alpha and beta values
    # algo = 'felzenszwalb'  # algorithm name
    # params = {'scale': 50, 'sigma': 1.5, 'min_size': 150}
    # shape = (84, 84, 3)  # data input shape
    # alpha = 150  # per-segment coalitions
    # beta = 0.15  # per-coalition active segments in % value

    # # Initialize sinex
    # support_input = Input(shape=(shot, 84, 84, 3))
    # query_input = Input(shape=(1, 84, 84, 3))
    # query_features = base_model.encoder(query_input[0])
    # x_flattened = tf.reshape(query_features, [1, -1])
    # support_features = base_model.encoder(
    #     tf.reshape(support_input, [shot, 84, 84, 3])
    # )
    # prototypes = tf.reduce_mean(
    #     tf.reshape(support_features, [shot, -1]),
    #     axis=1
    # )
    # dists = tf.norm(x_flattened - prototypes, axis=1)
    # model = Model(inputs=[support_input, query_input], outputs=-dists)
    # sinex = Sinex(algo, params, shape)
    # # Get SINEX explanations
    # E = sinex.explain(model, query.squeeze(0), episode_support_data)
    # plt = xai_plot(gradcam_target1_attributions, query[0])
    # # plt.savefig(f"./results/{dataset_str}_feature_attribution_map/gardcam_{target1_name}_Features_{input_model_str}_{shot}shot.png",dpi=450)
    # # Initialize SINEXC
    # sinexc = Sinexc(algo, params, shape, alpha, beta)
    # # Get SINEXC explanations
    # E = sinexc.explain(model, query.squeeze(0), episode_support_data)
