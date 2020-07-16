from predict import assess_models, assess_models_in_folders
from finetuning import finetuning_loop
from visualize_dataset import visualize_segmentation_dataset
#TODO
'''

'''

# train paths
train_frames_path = 'TissueDataset/Training/Original'
train_masks_path = 'TissueDataset/Training/Mask'
# train paths on compute canada machine
cc_train_frames_path = '../TissueDataset1024/Training/Original'
cc_train_masks_path =  '../TissueDataset1024/Training/Mask'

# validation paths
val_frames_path = 'TissueDataset/Validation/Original'
val_masks_path = 'TissueDataset/Validation/Mask'
# validation paths on compute canada machine
cc_val_frames_path = '../TissueDataset1024/Validation/Original'
cc_val_masks_path = '../TissueDataset1024/Validation/Mask'

# test paths
test_frames_path = 'TissueDataset/Test/Original'
test_masks_path = 'TissueDataset/Test/Mask'
# test paths on compute canada machine
cc_test_frames_path = '../TissueDataset1024/Test/Original'
cc_test_masks_path = '../TissueDataset1024/Test/Mask'

# output paths
history_dir = './history_dev/'
output_dir = './predictions_dev/'

# finetune the mini unet model
#finetuning_loop(history_dir, train_frames_path, train_masks_path, val_frames_path, val_masks_path, test_frames_path, test_masks_path)
finetuning_loop(history_dir, cc_train_frames_path, cc_train_masks_path, cc_val_frames_path, cc_val_masks_path, cc_test_frames_path, cc_test_masks_path)

# make predictions
#assess_models_in_folders(history_dir, test_frames_path, test_masks_path, output_dir)
#assess_models_in_folders(history_dir, cc_test_frames_path, cc_test_masks_path, output_dir)

#visualize dataset
#visualize_segmentation_dataset(train_frames_path, train_masks_path, 'test_aug')
