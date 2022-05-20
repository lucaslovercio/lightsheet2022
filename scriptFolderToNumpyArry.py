import os
import cv2
import numpy as np

folder_root = "/media/lucas/DATA/POSDOC_Calgary/Lightsheet/Image-to-Image/Annotations"

folder =        folder_root + "/Test/Obs1_Labels_192/Strips_192"
fileOutput =    folder_root + "/Test/Obs1_Labels_192/Strips_192/Test_Original_192_labels_Strips.npy"
sizeGAN = 192

items = sorted(os.listdir(folder)) #["frame_00", "frame_01", "frame_02", ...]

images_paths = []
for names in items:
    if names.endswith(".png"):
        images_paths.append(names)
print(images_paths)

imgs_array = []
for pathIn in images_paths:
    print(pathIn)
    fullpath = folder + "/" + pathIn
    print(fullpath)
    im = cv2.imread(fullpath, cv2.IMREAD_ANYDEPTH)
    print(im.shape)
    transformed_img = np.asarray(im)
    print(transformed_img.shape)
    transformed_img = cv2.resize(transformed_img, (sizeGAN, sizeGAN), interpolation=cv2.INTER_NEAREST)
    imgs_array.append(transformed_img) #.transpose(1, 0, 2))

imgs_array = np.array(imgs_array)
print(imgs_array.shape)

np.save(fileOutput, imgs_array)
