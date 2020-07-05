
import random

import numpy as np
import cv2

#from .augmentation import augment_seg
from .data_loader import image_segmentation_generator

random.seed(DATA_LOADER_SEED)


def _get_colored_segmentation_image(img, seg, colors, n_classes, do_augment=False):
    """ Return a colored segmented image """
    seg_img = np.zeros_like(seg)

    if do_augment:
        img, seg[:, :, 0] = augment_seg(img, seg[:, :, 0])

    for c in range(n_classes):
        seg_img[:, :, 0] += ((seg[:, :, 0] == c)
                             * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg[:, :, 0] == c)
                             * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg[:, :, 0] == c)
                             * (colors[c][2])).astype('uint8')

    return img, seg_img

#change segs and images to masks and frames
def visualize_segmentation_dataset(images_path, segs_path, n_classes, do_augment=False, ignore_non_matching=False, no_show=False):
    #TODO1 working
    plain_generator = image_segmentation_generator(#TODO1 add augmentation_type flags here
        images_path, segs_path,  1,  3,#batch_size = 1, num_classes = 3 TODO1 add proper path
        128, 128, norm_type=None)
    augmented_generator = image_segmentation_generator(#TODO1 add augmentation_type flags here
        images_path, segs_path,  1,  3,#batch_size = 1, num_classes = 3 TODO1 add proper path
        128, 128, norm_type=None)
    #TODO1 end working

        # Get the colors for the classes
        
        print("Please press any key to display the next image")
        while True:
            im, seg = next(plain_generator)
            print("Found the following classes in the segmentation image:", np.unique(seg))
            im_aug, seg_aug = next(augmented_generator)
            #TODO what does below do?
            img, seg_img = _get_colored_segmentation_image(
                                                    img, seg, colors,
                                                    n_classes,
                                                    do_augment=do_augment)
            print("Please press any key to display the next image")
            cv2.imshow("img", img)
            cv2.imshow("seg_img", seg_img)
            cv2.waitKey()
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--images", type=str)
#     parser.add_argument("--annotations", type=str)
#     parser.add_argument("--n_classes", type=int)
#     args = parser.parse_args()

#     visualize_segmentation_dataset(
#         args.images, args.annotations, args.n_classes)
 
