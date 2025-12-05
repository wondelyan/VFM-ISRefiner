import torch
import numpy as np
import skimage
import cv2
from .log import logger


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def save_checkpoint(net, checkpoints_path, epoch=None, prefix='', verbose=True, multi_gpu=False):
    if epoch is None:
        checkpoint_name = 'last_checkpoint.pth'
    else:
        checkpoint_name = f'{epoch:03d}.pth'

    if prefix:
        checkpoint_name = f'{prefix}_{checkpoint_name}'

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    if verbose:
        logger.info(f'Save checkpoint to {str(checkpoint_path)}')

    net = net.module if multi_gpu else net
    torch.save({'state_dict': net.state_dict(),
                'config': net._config}, str(checkpoint_path))


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax


def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    return (max(rmin, bbox[0]), min(rmax, bbox[1]),
            max(cmin, bbox[2]), min(cmax, bbox[3]))


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union


def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()



def get_instance_from_label_3bands(label, stuff_color_list):
    label_code = label[:, :, 0] + 255 * label[:, :, 1] + 255 * 255 * label[:, :, 2]

    labels, num_objects = skimage.measure.label(label_code, return_num=True)

    labels = labels + 1

    if stuff_color_list:
        for i, stuff_color in enumerate(stuff_color_list):
            labels = np.where(
                label_code == stuff_color[0] * stuff_color[1] * stuff_color[2], num_objects + 1 + i, labels)

    instance_list = set(labels.reshape(-1))

    return labels, list(instance_list)


def get_instance_from_label_1band(label, stuff_list=None, ignore_list=None):
    label = label.astype(np.int32)
    label_id = label[:, :, 0] * 256 * 256 + label[:, :, 1] * 256 + label[:, :, 2]

    if stuff_list != None:
        stuff_list = np.array(stuff_list).astype(np.int32)
        stuff_id_list = stuff_list[:, 0] * 256 * 256 + stuff_list[:, 1] * 256 + stuff_list[:, 2]
    else:
        stuff_id_list = []

    if ignore_list != None:
        ignore_list = np.array(ignore_list).astype(np.int32)
        ignore_id_list = ignore_list[:, 0] * 256 * 256 + ignore_list[:, 1] * 256 + ignore_list[:, 2]
    else:
        ignore_id_list = []

    label_id_list = list(set(label_id.flatten()))
    instances_id = np.zeros(label.shape[:2], dtype=np.int32)

    save_instance_id = 1
    save_instance_id_list = []
    for class_id in label_id_list:
        if class_id in stuff_id_list:
            binary_img = np.where(class_id == label_id, 255, 0).astype(np.uint8)
            num_instances, instances_label = cv2.connectedComponents(binary_img, connectivity=8)
            for id in range(1, num_instances):
                while save_instance_id in save_instance_id_list:
                    save_instance_id += 1
                instances_id = np.where(instances_label == id, save_instance_id, instances_id)
                save_instance_id_list.append(save_instance_id)
        elif class_id in ignore_id_list:
            pass
        else:
            while save_instance_id in save_instance_id_list:
                save_instance_id += 1
            instances_id = np.where(class_id == label_id, save_instance_id, instances_id)
            save_instance_id_list.append(save_instance_id)

    return instances_id, save_instance_id_list