import numpy as np
import cv2
from data_loader import image_segmentation_generator

#TODO make a kill key that ends the loop, maybe k
def visualize_segmentation_dataset(frames_path, masks_path, aug_type):
    plain_generator = image_segmentation_generator(
        frames_path, masks_path,  1,  3,#batch_size = 1, num_classes = 3
        128, 128, norm_type=None, deterministic=True)
    augmented_generator = image_segmentation_generator(
        frames_path, masks_path,  1,  3,#batch_size = 1, num_classes = 3
        128, 128, norm_type=None, aug_type=aug_type, deterministic=True)
        
    while True:
        img, mask = next(plain_generator)
        img_aug, mask_aug = next(augmented_generator)
        print("Please press any key to display the next image")
        cv2.waitKey(0)
        img = img[0,:,:,0].astype(np.uint8)
        mask = np.argmax(mask[0,:,:,:], axis=-1).astype(np.uint8) * 50
        img_aug = img_aug[0,:,:,0].astype(np.uint8)
        mask_aug = np.argmax(mask_aug[0,:,:,:], axis=-1).astype(np.uint8) * 50
        cv2.imshow('Dataset', np.concatenate((np.concatenate((img, mask), axis=1), np.concatenate((img_aug, mask_aug), axis=1)), axis=0))
