from segment_image import assess_models, assess_models_in_folders
from finetuning import finetuning_loop
from visualize_dataset import visualize_segmentation_dataset#TODO1 new
#TODO
'''

'''

# finetune the mini unet model
train_frames_path = 'TissueDataset/Training/Original'
train_masks_path = 'TissueDataset/Training/Mask'

val_frames_path = 'TissueDataset/Validation/Original'
val_masks_path = 'TissueDataset/Validation/Mask'
history_dir = 'history_dev/'

#finetuning_loop(history_dir, train_frames_path, train_masks_path, val_frames_path, val_masks_path)

# make predictions
test_frame_path = 'TissueDataset/Test/Original'
test_mask_path = 'TissueDataset/Test/Mask'
output_dir = 'predictions_dev/'

#assess_models(history_dir, test_frame_path, test_mask_path, output_dir)#deprecated
assess_models_in_folders(history_dir, test_frame_path, test_mask_path, output_dir)

#visualize dataset
#visualize_segmentation_dataset(train_frames_path, train_masks_path, 'all')

