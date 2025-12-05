import cv2
import json
import random
import numpy as np
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from pycocotools.coco import COCO


class SSDDDataset(ISDataset):
    def __init__(self, dataset_path, **kwargs):
        super(SSDDDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.dataset = COCO(self.dataset_path / 'SSDD.json')
        self.load_samples()

    def load_samples(self):
        self.dataset_samples = self.dataset.getImgIds()

    def get_sample(self, index) -> DSample:
        img_id = self.dataset_samples[index]
        image_dir = self.dataset.loadImgs(img_id)[0]

        image = cv2.imread(str(self.dataset_path / 'SSDD' / image_dir['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instance_id_list = self.dataset.getAnnIds(imgIds=img_id)
        instance_map = np.full((image.shape[0], image.shape[1]), 0)
        for instance_id in instance_id_list:
            instance = self.dataset.loadAnns(instance_id)[0]
            instance_map[self.dataset.annToMask(instance) == 1] = instance_id

        return DSample(image, instance_map, objects_ids=instance_id_list)
