import os
import numpy as np
from PIL import Image

class DataLoader(object):
    def __init__(self, data_dir, split, n_way, n_support, n_query):
        self.data_dir = data_dir
        self.split = split
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.split_file = os.path.join(data_dir, 'splits', 'default', f'{split}.csv')
        self.img_paths, self.labels = self._load_split()
        self.unique_labels = list(set(self.labels))
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.n_classes = len(self.unique_labels)

    def _load_split(self):
        with open(self.split_file, 'r') as f:
            lines = f.readlines()[1:]  # 跳过第一行标题行
            filenames = [line.split(',')[0] for line in lines]
            labels = [line.split(',')[1].strip() for line in lines]
        img_paths = [os.path.join(self.data_dir, 'data', f"{filename}") for filename in filenames]
        return img_paths, labels

    def get_next_episode(self):
        support = np.zeros([self.n_way, self.n_support, 84, 84, 3], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, 84, 84, 3], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            class_label = self.unique_labels[i_class]
            class_indices = [idx for idx, label in enumerate(self.labels) if label == class_label]
            selected = np.random.permutation(class_indices)[:self.n_support + self.n_query]

            for j, idx in enumerate(selected[:self.n_support]):
                support[i, j] = self._load_and_preprocess_image(self.img_paths[idx])

            for j, idx in enumerate(selected[self.n_support:]):
                query[i, j] = self._load_and_preprocess_image(self.img_paths[idx])

        return support, query

    def _load_and_preprocess_image(self, img_path):
        img = Image.open(img_path).resize((84, 84))
        img = np.asarray(img) / 255.0
        # img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return img

def load_dogs(data_dir, config, splits):
    """
    Load Standford Dogs dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as DataLoader
    """
    ret = {}
    for split in splits:
        if split in ['val', 'test']:
            n_way = config['data.test_way']
            n_support = config['data.test_support']
            n_query = config['data.test_query']
        else:
            n_way = config['data.train_way']
            n_support = config['data.train_support']
            n_query = config['data.train_query']

        data_loader = DataLoader(data_dir, split, n_way, n_support, n_query)
        ret[split] = data_loader

    return ret