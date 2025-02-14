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
from miniImagenet_dogs_cars_birds import print_dict_info, print_ndarray_info, resize_the_batch, model_wrapper
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

import numpy as np
import tensorflow as tf


def calculate_iAUC(model, feature_attribution_map, input_image, query):
    # 初始化一个空的输入（全黑的频谱图）
    empty_input = np.zeros_like(input_image)

    # 获取像素的重要性排序
    sorted_indices = np.argsort(feature_attribution_map, axis=None)[::-1]

    # 逐步插入像素并记录模型输出
    model_outputs = []
    for i in range(len(sorted_indices)):
        # 获取当前像素的坐标
        idx = np.unravel_index(sorted_indices[i], input_image.shape)

        # 插入像素
        empty_input[idx] = input_image[idx]

        # 将输入转换为模型所需的格式
        support_input = np.expand_dims(empty_input, axis=0)
        query_input = np.expand_dims(query, axis=0)

        # 获取模型输出
        output = model([support_input, query_input])
        model_outputs.append(output[0])

    model_outputs = normalize_list(model_outputs)
        # 计算iAUC
    iAUC = np.trapz(model_outputs, dx=1.0 / len(sorted_indices))

    return iAUC, model_outputs


def calculate_dAAC(model, feature_attribution_map, input_image, query):
    # 初始化一个完整的输入（原始的频谱图）
    full_input = np.copy(input_image)

    # 获取像素的重要性排序
    sorted_indices = np.argsort(feature_attribution_map, axis=None)[::-1]

    # 逐步删除像素并记录模型输出
    model_outputs = []
    for i in range(len(sorted_indices)):
        # 获取当前像素的坐标
        idx = np.unravel_index(sorted_indices[i], input_image.shape)

        # 删除像素
        full_input[idx] = 0

        # 将输入转换为模型所需的格式
        support_input = np.expand_dims(full_input, axis=0)
        query_input = np.expand_dims(query, axis=0)

        # 获取模型输出
        output = model([support_input, query_input])
        model_outputs.append(output[0])

    model_outputs = normalize_list(model_outputs)
        # 计算dAAC
    dAAC = np.trapz(model_outputs, dx=1.0 / len(sorted_indices))

    return dAAC, model_outputs

def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst

def normalize_feature_attribution(feature_attributions, input_percentile=99):
    # 归一化处理
    abs_vals = np.abs(feature_attributions)
    max_val = np.nanpercentile(abs_vals, input_percentile)

    vmin = -max_val
    vmax = max_val

    normalized = (feature_attributions - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)  # 确保值在0到1之间

    return normalized


def evaluate_explanation(model, feature_attribution_map, input_image, query, input_percentile=99):
    # 归一化feature_attribution_map
    normalized_attribution = normalize_feature_attribution(feature_attribution_map, input_percentile)

    # 计算iAUC和插入曲线
    iAUC, insertion_curve = calculate_iAUC(model, normalized_attribution, input_image, query)

    # 计算dAAC和删除曲线
    dAAC, deletion_curve = calculate_dAAC(model, normalized_attribution, input_image, query)

    return iAUC, dAAC, insertion_curve, deletion_curve

def plot_curves(insertion_curve, deletion_curve, save_path=None):
    # 设置图像尺寸为长方形（宽度大于高度）
    plt.figure(figsize=(10, 2))

    # 归一化x轴到[0, 1]范围
    x = np.linspace(0, 1, len(insertion_curve))

    # 绘制插入曲线
    plt.subplot(1, 2, 1)
    plt.plot(x, insertion_curve, label='Insertion Curve')
    plt.fill_between(x, insertion_curve, color='blue', alpha=0.1)
    plt.xlabel('Pixels inserted')
    plt.legend()
    plt.grid(color='gray', linewidth=0.5, alpha=0.5)  # 添加灰色网格线
    for y_val in [0.2, 0.4, 0.6, 0.8]:
        plt.axhline(y=y_val, color='gray', linewidth=0.5, alpha=0.5)
    # 绘制删除曲线
    plt.subplot(1, 2, 2)
    plt.plot(x, deletion_curve, label='Deletion Curve')
    plt.fill_between(x, deletion_curve, color='blue', alpha=0.1)
    plt.xlabel('Pixels inserted')
    plt.legend()
    plt.grid(color='gray', linewidth=0.5, alpha=0.5)  # 添加灰色网格线
    for y_val in [0.2, 0.4, 0.6, 0.8]:
        plt.axhline(y=y_val, color='gray', linewidth=0.5, alpha=0.5)

    # 调整子图间距
    plt.tight_layout()

    # 如果提供了保存路径，则保存图像
    if save_path:
        # 保存为矢量图
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

        # 显示图像
    plt.show()

if __name__ == '__main__':

    ### This few-shot XAI framwork need you to specify shot number.
    shot = 1
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
    save_dir = f"./results/{dataset_str}_episodic_xai"
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

    class_idx, class_num = 0, 5
    ### SINEXC XAI
    # Set algorithms parameters, input data shape, alpha and beta values
    algo = 'felzenszwalb'  # algorithm name
    params = {'scale': 50, 'sigma': 1.5, 'min_size': 150}
    shape = (84, 84, 3)  # data input shape
    alpha = 150  # per-segment coalitions
    beta = 0.15  # per-coalition active segments in % value

    # Initialize sinex
    support_input = Input(shape=(shot, 84, 84, 3))
    query_input = Input(shape=(1, 84, 84, 3))
    query_features = base_model.encoder(query_input[0])
    x_flattened = tf.reshape(query_features, [1, -1])
    support_features = base_model.encoder(
        tf.reshape(support_input, [shot, 84, 84, 3])
    )
    prototypes = tf.reduce_mean(
        tf.reshape(support_features, [shot, -1]),
        axis=0
    )
    dists = tf.norm(x_flattened - prototypes, axis=1)
    model = Model(inputs=[support_input, query_input], outputs=-dists)
    sinex = Sinex(algo, params, shape)
    # Get SINEX explanations
    # E = sinex.explain(model, query.squeeze(0), episode_support_data)
    # for i in range(len(E)):
    #     plt = xai_plot(E[i], episode_support_data[i][0])
    #     plt.savefig(f"./results/{dataset_str}_episodic_xai/sinex_{i}_Features_{input_model_str}_{shot}shot.png",dpi=450)
    # # 假设model, feature_attribution_map, input_image, class_idx已经定义
    # sinex_iAUC, sinex_dAAC, sinex_insertion_curve, sinex_deletion_curve = evaluate_explanation(model, E[class_idx], episode_support_data[class_idx], query)
    # plot_curves(sinex_insertion_curve, sinex_deletion_curve, save_path=f'./results/{dataset_str}_episodic_xai/sinex_curves_{input_model_str}_{shot}shot.svg')
    # print(f'Eposidic SINEX: iAUC->{sinex_iAUC}, dAAC->{sinex_dAAC}.')
    #
    # # Initialize SINEXC
    # sinexc = Sinexc(algo, params, shape, alpha, beta)
    # # Get SINEXC explanations
    # Ec = sinexc.explain(model, query.squeeze(0), episode_support_data)
    # for i in range(len(Ec)):
    #     plt = xai_plot(Ec[i], episode_support_data[i][0])
    #     plt.savefig(f"./results/{dataset_str}_episodic_xai/sinexc_{i}_Features_{input_model_str}_{shot}shot.png",dpi=450)
    # sinexc_iAUC, sinexc_dAAC, sinexc_insertion_curve, sinexc_deletion_curve = evaluate_explanation(model, Ec[class_idx], episode_support_data[class_idx], query)
    # plot_curves(sinexc_insertion_curve, sinexc_deletion_curve, save_path=f'./results/{dataset_str}_episodic_xai/sinex_curves_{input_model_str}_{shot}shot.svg')
    # print(f'Eposidic SINEXC: iAUC->{sinexc_iAUC}, dAAC->{sinexc_dAAC}.')

    # ### C3A XAI
    c3a_xai = C3Amodel(base_model.encoder, feature_layer=feature_layer, k=k)
    c3a_target_scores = []
    for i in range(class_num):
        exclude_data = np.concatenate((episode_support_data[:i, :, :, :, :], episode_support_data[i+1:, :, :, :, :]), axis=0)
        c3a_target_scores.append(c3a_xai.image_feature_attribution_c3a(
            support_data_1=query,
            support_data_2=exclude_data.squeeze(1),
            support_data_3=exclude_data.squeeze(1),
            query=episode_support_data[i], ref_pixel=ref_pixel, pad=padding_size))
    for i in range(len(c3a_target_scores)):
        plt = xai_plot(c3a_target_scores[i], episode_support_data[i][0])
        plt.savefig(f"./results/{dataset_str}_episodic_xai/c3a_{i}_Features_{input_model_str}_{shot}shot.png",dpi=450)
    c3a_iAUC, c3a_dAAC, c3a_insertion_curve, c3a_deletion_curve = evaluate_explanation(model, c3a_target_scores[class_idx], episode_support_data[class_idx], query)
    plot_curves(c3a_insertion_curve, c3a_deletion_curve, save_path=f'./results/{dataset_str}_episodic_xai/c3a_curves_{input_model_str}_{shot}shot.svg')
    print(f'Eposidic C3A: iAUC->{c3a_iAUC}, dAAC->{c3a_dAAC}.')

    ### ProtoShotXAI
    protoshot_xai = ProtoShotXAI(base_model.encoder, feature_layer=flatten_layer)
    protoshot_target_scores = []
    for i in range(class_num):
        protoshot_target_scores = protoshot_xai.image_feature_attribution(
            support_data=query, query=episode_support_data[i], ref_pixel=ref_pixel,
            pad=padding_size
        )
    for i in range(len(protoshot_target_scores)):
        plt = xai_plot(protoshot_target_scores[i], episode_support_data[i][0])
        plt.savefig(f"./results/{dataset_str}_episodic_xai/protoshot_{i}_Features_{input_model_str}_{shot}shot.png",dpi=450)
    protoshot_iAUC, protoshot_dAAC, protoshot_insertion_curve, protoshot_deletion_curve = evaluate_explanation(model, protoshot_target_scores[class_idx], episode_support_data[class_idx], query)
    plot_curves(protoshot_insertion_curve, protoshot_deletion_curve, save_path=f'./results/{dataset_str}_episodic_xai/protoshot_curves_{input_model_str}_{shot}shot.svg')
    print(f'Eposidic ProtoShotXAI: iAUC->{protoshot_iAUC}, dAAC->{protoshot_dAAC}.')

    ## GradCAM++ XAI
    # Find last conv laver (Use pre-determined layers instead).
    last_conv_layer_name = base_model.encoder.layers[layer_num].name
    gradcam_target_scores = []
    gradcam = GradCam(base_model.encoder, last_conv_layer_name, query)
    for i in range(class_num):
        gradcam_target_scores = gradcam.make_gradcam_heatmap(np.expand_dims(episode_support_data[i], axis=0), pred_index=0)
    for i in range(len(gradcam_target_scores)):
        plt = xai_plot(gradcam_target_scores[i], episode_support_data[i][0])
        plt.savefig(f"./results/{dataset_str}_episodic_xai/gradcam_{i}_Features_{input_model_str}_{shot}shot.png",dpi=450)
    gradcam_iAUC, gradcam_dAAC, gradcam_insertion_curve, gradcam_deletion_curve = evaluate_explanation(model, gradcam_target_scores[class_idx], episode_support_data[class_idx], query)
    plot_curves(gradcam_insertion_curve, gradcam_deletion_curve, save_path=f'./results/{dataset_str}_episodic_xai/gradcam_curves_{input_model_str}_{shot}shot.svg')
    print(f'Eposidic GradCAM++: iAUC->{gradcam_iAUC}, dAAC->{gradcam_dAAC}.')