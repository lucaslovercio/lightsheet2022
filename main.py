from segment_image import assess_models
from finetuning import finetuning_loop
#TODO
'''

'''

# finetune the mini unet model
train_frames_path = 'TissueDataset/Training/Original'
train_masks_path = 'TissueDataset/Training/Mask'

val_frames_path = 'TissueDataset/Validation/Mask'
val_masks_path = 'TissueDataset/Validation/Mask'
history_dir = 'history_dev/'

finetuning_loop(history_dir, train_frames_path, train_masks_path, val_frames_path, val_masks_path)

# make predictions
test_frame_path = val_frames_path #'TissueDataset/Test/Original'
test_mask_path = val_masks_path #'TissueDataset/Test/Mask'
output_dir = 'predictions_dev/'

assess_models(history_dir, test_frame_path, test_mask_path, output_dir)
