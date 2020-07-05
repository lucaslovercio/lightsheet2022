import numpy as np
import cv2
from data_loader import image_segmentation_generator

#from .augmentation import augment_seg


#change segs and images to masks and frames
def visualize_segmentation_dataset(images_path, segs_path, n_classes, do_augment=False, ignore_non_matching=False, no_show=False):
    plain_generator = image_segmentation_generator(
        images_path, segs_path,  1,  3,#batch_size = 1, num_classes = 3 TODO1 add proper path
        128, 128, norm_type=None, deterministic=True)
    augmented_generator = image_segmentation_generator(
        images_path, segs_path,  1,  3,#batch_size = 1, num_classes = 3 TODO1 add proper path
        128, 128, norm_type=None, deterministic=True)#TODO1 add augmentation_type flags here
        
    while True:
        im, seg = next(plain_generator)
        im_aug, seg_aug = next(augmented_generator)
        print("Please press any key to display the next image")
        cv2.waitKey(0)
        im = im[0,:,:,0].astype(np.uint8)
        seg = np.argmax(seg[0,:,:,:], axis=-1).astype(np.uint8) * 50
        im_aug = im_aug[0,:,:,0].astype(np.uint8)
        seg_aug = np.argmax(seg_aug[0,:,:,:], axis=-1).astype(np.uint8) * 50
        cv2.imshow('originals', np.concatenate((np.concatenate((im, seg), axis=1), np.concatenate((im_aug, seg_aug), axis=1)), axis=0))
