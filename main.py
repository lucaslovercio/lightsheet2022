from predict import assess_models, assess_models_in_folders, segment_folder
from finetuning import finetuning_loop, finetuning_random
from visualize_dataset import visualize_segmentation_dataset
import tensorflow as tf
#TODO
'''
'''

# whether or not this is being run on the compute canada machine
cc = False

def main():

    #Is TF using GPU?
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    folderInput = '/media/lucas/DATA/POSDOC_Calgary/Lightsheet/Image-to-Image/Annotations/'
    # output paths
    history_dir = '/media/lucas/DATA/POSDOC_Calgary/Lightsheet/Image-to-Image/Annotations/history_only_good/'
    output_dir = './predictions_dev/'
    if cc:
        # train paths
        train_frames_path = '../TissueDataset1024/Training/Original'
        train_masks_path = '../TissueDataset1024/Training/Mask'
        # validation paths
        val_frames_path = '../TissueDataset1024/Validation/Original'
        val_masks_path = '../TissueDataset1024/Validation/Mask'
        # test paths
        test_frames_path = '../TissueDataset1024/Test/Original'
        test_masks_path = '../TissueDataset1024/Test/Mask'
        # training specifications
        image_size = 1024
        max_epochs = 250
        patience = 20
    else:
        # train paths
        train_frames_path = folderInput + 'Training/Original_192/Good_192'
        train_masks_path = folderInput + 'Training/Obs_Labels_192/Good_192'
        # validation paths
        val_frames_path = folderInput + 'Validation/Original_192/Good_192'
        val_masks_path = folderInput + 'Validation/Obs_Labels_192/Good_192'
        # test paths
        #test_frames_path = 'TissueDataset/Test/Original'
        #test_masks_path = 'TissueDataset/Test/Mask'
        # training specifications
        image_size = 192
        max_epochs = 350
        patience = 20
        #TODO1 changed below
        test_frames_path = folderInput + 'Test/Original_192/Good_192'
        test_masks_path = folderInput + 'Test/Obs1_Labels_192/Good_192'

    #finetune the mini unet model
    # finetuning_random(history_dir, train_frames_path, train_masks_path, val_frames_path, val_masks_path, test_frames_path, test_masks_path, 
    #                   image_size, max_epochs, patience,
    #                   num_models=10)
    finetuning_loop(history_dir, train_frames_path, train_masks_path,
                    val_frames_path, val_masks_path,
                    test_frames_path, test_masks_path,
                    image_size, max_epochs, patience)

    #make predictions
    # assess_models_in_folders(history_dir, test_frames_path, test_masks_path, output_dir)

    #segment a folder of images with a specified model
    #segment_folder(history_dir + 'unetMini_num_0_normtype_divide/unetMini_num_0_normtype_divide.h5', test_frames_path, output_dir)
    #segment_folder(history_dir + 'unetMini_num_0_normtype_divide/unetMini_num_0_normtype_divide.h5', '../VolumeDAPITiles_ForTissue', '../VolumeDAPISegs_ForTissue')
    #visualize dataset
    # visualize_segmentation_dataset(train_frames_path, train_masks_path, 'distortionless')

main()
