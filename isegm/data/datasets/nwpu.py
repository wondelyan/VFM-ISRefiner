import cv2
import json
import random
import numpy as np
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from pycocotools.coco import COCO


class NWPUDataset(ISDataset):
    def __init__(self, dataset_path, **kwargs):
        super(NWPUDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.labels_path = None
        self.images_path = None
        self.instanse_list = None
        self.load_samples()
    def load_samples(self):
        self.labels_path = self.dataset_path / 'ann_dir_rgb'
        self.images_path = self.dataset_path / 'img_dir_png'

        self.dataset_samples = [x.stem for x in self.labels_path.iterdir() if x.suffix == '.png']

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]

        image_path = self.images_path / (dataset_sample + '.png')
        label_path = self.labels_path / (dataset_sample + '.png')
        image = cv2.imread(str(image_path))
        
        instance_map = cv2.imread(str(label_path))
        instance_map_id = instance_map[:, :, 0] + 255 * instance_map[:, :, 1] + 255 * 255 * instance_map[:, :, 2]
        instances_ids = list(set(instance_map_id.reshape(-1)))
        if 0 in instances_ids:
            instances_ids.remove(0)

        return DSample(image, instance_map_id, objects_ids=instances_ids)