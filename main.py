from predict import assess_models, assess_models_in_folders
from finetuning import finetuning_loop, finetuning_random
from visualize_dataset import visualize_segmentation_dataset
#TODO
'''
'''

# whether or not this is being run on the compute canada machine
cc = False

def main():
    # output paths
    history_dir = './history_dev/'
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
        max_epochs = 500
        patience = 25
    else:
        # train paths
        train_frames_path = 'TissueDataset/Training/Original'
        train_masks_path = 'TissueDataset/Training/Mask'
        # validation paths
        val_frames_path = 'TissueDataset/Validation/Original'
        val_masks_path = 'TissueDataset/Validation/Mask'
        # test paths
        test_frames_path = 'TissueDataset/Test/Original'
        test_masks_path = 'TissueDataset/Test/Mask'
        # training specifications
        image_size = 128
        max_epochs = 150
        patience = 10

    # finetune the mini unet model
    finetuning_random(history_dir, train_frames_path, train_masks_path, val_frames_path, val_masks_path, test_frames_path, test_masks_path, 
                      image_size, max_epochs, patience,
                      num_models=10)
    # finetuning_loop(history_dir, train_frames_path, train_masks_path,
    #                 val_frames_path, val_masks_path,
    #                 test_frames_path, test_masks_path,
    #                 image_size, max_epochs, patience)

    # make predictions
    assess_models_in_folders(history_dir, test_frames_path, test_masks_path, output_dir)
    
    #visualize dataset
    #visualize_segmentation_dataset(train_frames_path, train_masks_path, 'distortionless')

main()
