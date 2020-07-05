import itertools
import os
import random
import numpy as np
import cv2
from augmentation import augment_pair
#TODO:
'''
- once there's a decision about 8-bit vs 16-bit the division norm needs to be changed
- add documentation to this file
'''


class DataLoaderError(Exception):
    pass


def get_pairs_from_paths(images_path, segs_path):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp"]

    image_files = []
    segmentation_files = {}

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry)))

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
           os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(segs_path, dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0}"
                                      " already exists and is ambiguous to"
                                      " resolve with path {1}."
                                      " Please remove or rename the latter."
                                      .format(file_name, full_dir_entry))
            segmentation_files[file_name] = (file_extension, full_dir_entry)

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            return_value.append((image_full_path,
                                segmentation_files[image_file][1]))
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation "
                                  "found for image {0}."
                                  .format(image_full_path))

    return return_value

# really as this method does is normalize and add a dimension, not load
def get_image_array(img, norm_type):
    """ Load image array from input """
    img = img.astype(np.float32)
    # normalization by division assumes that the input images were 16-bit
    if norm_type == 'divide':
        img /= 65536.0
    elif norm_type == 'sub_mean':
        img -= np.mean(img)
    elif norm_type == 'divide_and_sub':
        img /= 65536.0
        img -= np.mean(img)

    img = img.reshape(img.shape + (1,))

    return img

#TODO change nClasses here to n_classes
# really all this does is convert the segmentations to one-hot encodings
def get_segmentation_array(img, nClasses, width, height):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    for c in range(nClasses):#TODO seems this could be done simpler
        seg_labels[:, :, c] = (img == c).astype(int)

    return seg_labels

# TODO change im and seg to img and mask
def image_segmentation_generator(images_path, segs_path, batch_size, n_classes, output_height, output_width,
                                 norm_type=None, aug_type=None, deterministic=False):

    #TODO try running the dataset visualizer w/o this part, see what happens
    if deterministic:
        random.seed(0)

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)

            im = cv2.imread(im, cv2.IMREAD_ANYDEPTH)#new changed from 0
            seg = cv2.imread(seg, cv2.IMREAD_ANYDEPTH)#new changed from 0

            # apply augmentation
            if aug_type is not None:
                im, seg = augment_pair(im, seg, aug_type)
            
            X.append(get_image_array(im, norm_type))
            Y.append(get_segmentation_array(seg, n_classes, output_width, output_height))
            
        yield np.array(X), np.array(Y)

# new (this function is designed to feed segmentImage.prediction in batches so the normalization works:
# this function is now deprecated
def image_generator(images_path, batch_size):
    image_files = []

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append(os.path.join(images_path, dir_entry))

    zipped = itertools.cycle(image_files)
    while True:
        X = []
        for _ in range(batch_size):
            im = next(zipped)
            im = cv2.imread(im, 0)
            X.append(get_image_array(im))

        yield np.array(X)
