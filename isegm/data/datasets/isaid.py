import cv2
import random
import numpy as np
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.misc import get_instance_from_label_3bands


class iSAIDDataset(ISDataset):
    def __init__(self, dataset_path, split='train', instanse_mode='instance', unlabel_prob=0.3, **kwargs):
        super(iSAIDDataset, self).__init__(**kwargs)
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.labels_path = None
        self.images_path = None
        self.instanse_mode = instanse_mode
        self.unlabel_prob = unlabel_prob
        self.not_stuff_list = None
        self.load_samples()

    def load_samples(self):
        if self.instanse_mode == 'instance':
            self.labels_path = self.dataset_path / self.split / 'Instance_masks' # / 'images'
        if self.instanse_mode == 'class':
            self.labels_path = self.dataset_path / self.split / 'Semantic_masks' # / 'images'
        self.images_path = self.dataset_path / self.split / 'images' # / 'images'
        if self.instanse_mode == 'instance':
            self.dataset_samples = [x.stem[:-16] for x in self.labels_path.iterdir() if x.suffix == '.png']
        elif self.instanse_mode == 'class':
            self.dataset_samples = [x.stem[:-19] for x in self.labels_path.iterdir() if x.suffix == '.png']
        else:
            raise ValueError('Unknown instanse_mode')

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]

        image_path = self.images_path / (dataset_sample + '.png')
        if self.instanse_mode == 'instance':
            label_path = self.labels_path / (dataset_sample + '_instance_id_RGB.png')
        elif self.instanse_mode == 'class':
            label_path = self.labels_path / (dataset_sample + '_instance_color_RGB.png')
        else:
            raise ValueError('Unknown instanse_mode')
        image = cv2.imread(str(image_path))
        instance_map = cv2.imread(str(label_path))
        instance_map_id = instance_map[:, :, 0] + 255 * instance_map[:, :, 1] + 255 * 255 * instance_map[:, :, 2]
        instances_ids = list(set(instance_map_id.reshape(-1)))

        if 0 in instances_ids:
            instances_ids.remove(0)

        return DSample(image, instance_map_id, objects_ids=instances_ids)
