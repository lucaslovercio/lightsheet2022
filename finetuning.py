import os
from keras.callbacks import EarlyStopping
import numpy as np

from data_loader import image_segmentation_generator
from unet_mini import unet
from save_training import save_model
#TODO
'''
- add proper documentation to this function
'''

# global variables
IMG_SIZE = 128 # replace with 1024 to run on fullsize images
MAX_EPOCHS = 300 # replace with something >> 300 for compute canada
PATIENCE = 10 # train for this many epochs without improvement replace with ~50 or ~100 for compute canada
MONITOR = 'val_loss' # monitor this for early stopping
OPTIM_TYPE = 'min' # either min or max, depending on MONITOR

# hyperparameters
BATCH_SIZES = [8]
LEARNING_RATES = [1e-2, 1e-3, 1e-4, 1e-5] # replace with [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
LOSSES = ['categorical_crossentropy', 'dice20_cce80', 'dice50_cce50', 'dice']
ACTIVATIONS = ['relu', 'sigmoid', 'tanh'] # replace with ['relu', 'sigmoid', 'tanh']
ACTIVATION_LASTS = ['softmax']
MAXPOOLINGS = [2,4]
FIRST_FILTERS = [8,16,32,64]
KERNEL_SIZES = [3,7,15]
DROPOUT = [True,False]
BATCH_NORM = [True,False]
NORM_TYPES = ['divide'] # replace with [None, 'divide', 'sub_mean', 'divide_and_sub']
OPTIMIZERS = ['adam', 'rmsprop']
AUGMENTATIONS = [None]# replace with [None, 'distortionless']



def finetuning_loop(history_dir, train_frames_path, train_masks_path, val_frames_path, val_masks_path):
    # variable to store the highest recorded f1 score to date
    best_f1 = -1
    # variable to track how many models have been trained so far
    counter = 0
    # grid search over hyperparameters
    for maxpool in MAXPOOLINGS:
        for activation in ACTIVATIONS:
            for activation_last in ACTIVATION_LASTS:
                for batch_size in BATCH_SIZES:
                    for norm_type in NORM_TYPES:
                        for loss in LOSSES:
                            for first_filters in FIRST_FILTERS:
                                for learning_rate in LEARNING_RATES:
                                    for kernel_size in KERNEL_SIZES:
                                        for dp in DROPOUT:
                                            for batch_norm in BATCH_NORM:
                                                for optimizer in OPTIMIZERS:
                                                    for augmentation in AUGMENTATIONS:
                                                        
                                                        # create generators for training and validation images
                                                        train_generator = image_segmentation_generator(
                                                            train_frames_path, train_masks_path,  batch_size,  3,
                                                            128, 128, norm_type, aug_type=augmentation)
                                                        val_generator = image_segmentation_generator(
                                                            val_frames_path, val_masks_path,  batch_size,  3,
                                                            128, 128, norm_type, aug_type=augmentation)
                                                        
                                                        num_train_images = len(os.listdir(train_frames_path ))
                                                        num_val_images = len(os.listdir(val_frames_path))
                                                        
                                                        # build the model
                                                        modelUnet = unet(lr = learning_rate, input_size = (IMG_SIZE, IMG_SIZE,1), loss_mode = loss,
                                                                         firstFilters = first_filters, kSize = kernel_size,
                                                                         activation_last=activation_last, pool_size_max_pooling=maxpool, batchNorm = batch_norm,
                                                                         dropOutLayerFlag=dp, activation=activation, optimizer=optimizer)

                                                        # train the model
                                                        # print out the counter recording how many models have been trained in this loop
                                                        counter += 1
                                                        print('Now training model', counter)
                                                        # from https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
                                                        es = EarlyStopping(monitor=MONITOR, mode=OPTIM_TYPE, verbose=1, patience=PATIENCE, restore_best_weights=True)
                                                        results = modelUnet.fit_generator(train_generator, epochs=MAX_EPOCHS,
                                                                                          steps_per_epoch = (num_train_images//batch_size),
                                                                                          validation_data=val_generator,
                                                                                          validation_steps=(num_val_images//batch_size),verbose=0,
                                                                                          callbacks = [es])
                                                        
                                                        # save the model
                                                        model_name = str(modelUnet.name) \
                                                            +  '_loss_' + str(loss) \
                                                            + '_filters_' + str(first_filters) \
                                                            + '_lr_' + str(learning_rate) \
                                                            + '_activation_' + str(activation) \
                                                            + '_ksize_' + str(kernel_size) \
                                                            + '_activation_last_' + str(activation_last) \
                                                            + '_maxpool_' + str(maxpool) \
                                                            + '_batchnorm_' + str(batch_norm)\
                                                            + '_dropout_' + str(dp) \
                                                            + '_optim_' + str(optimizer) \
                                                            + '_aug_' + str(augmentation) \
                                                            + '_normtype_' + str(norm_type)# make sure norm_type is part of the name, or assess_model() won't work
                                                        # total number of epochs this model was trained for
                                                        last_epoch = len(results.history[MONITOR]) - 1 # note that the first epoch is "0"
                                                        # number of epochs before early stopping saved the best model
                                                        best_model_epoch = last_epoch - PATIENCE
                                                        # the best F1 score achieved while training this model
                                                        current_f1 = results.history['val_f1_macro'][best_model_epoch]
                                                        # if the current model has the best F1 score yet, save it
                                                        if current_f1 > best_f1:
                                                            best_f1 = current_f1
                                                            save_model(modelUnet, results, last_epoch, best_model_epoch, model_name, history_dir)
