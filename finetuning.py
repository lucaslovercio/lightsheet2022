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
PATIENCE = 20 # train for this many epochs without improvement replace with ~50 or ~100 for compute canada
MONITOR = 'val_loss' # monitor this for early stopping
OPTIM_TYPE = 'min' # either min or max, depending on MONITOR

# hyperparameters
BATCH_SIZES = [16]
LEARNING_RATES = [1e-2, 1e-4] # replace with [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
LOSSES = ['categorical_crossentropy', 'dice20_cce80'] # replace with ['categorical_crossentropy', 'dice20_cce80', 'dice50_cce50', 'dice']
ACTIVATIONS = ['relu'] # replace with ['relu', 'sigmoid', 'tanh']
ACTIVATION_LASTS = ['softmax']
MAXPOOLINGS = [2,4]
FIRST_FILTERS = [8, 16] # replace with [8, 16, 32, 64]
KERNEL_SIZES = [3,7,15]
DROPOUT = [True,False]
BATCH_NORM = [True,False]
NORM_TYPES = ['divide'] # replace with [None, 'divide', 'sub_mean', 'divide_and_sub']
OPTIMIZERS = ['adam'] # replace with ['adam', 'rmsprop', 'SGD']
AUGMENTATIONS = [None]# replace with [None, 'distortionless']



def finetuning_loop(history_dir, train_frames_path, train_masks_path, val_frames_path, val_masks_path, test_frames_path, test_masks_path):#new test paths
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
                                                            IMG_SIZE, IMG_SIZE, norm_type, aug_type=augmentation)
                                                        val_generator = image_segmentation_generator(
                                                            val_frames_path, val_masks_path,  batch_size,  3,
                                                            IMG_SIZE, IMG_SIZE, norm_type)
                                                        
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
                                                        results = modelUnet.fit(train_generator,
                                                                                epochs=MAX_EPOCHS,
                                                                                steps_per_epoch = (num_train_images//batch_size),
                                                                                validation_data=val_generator,
                                                                                validation_steps=(num_val_images//batch_size),verbose=0,
                                                                                #use_multiprocessing=True, #TODO1
                                                                                callbacks = [es])
                                                        
                                                        # evaluate the model (new) (for debugging)
                                                        va = modelUnet.evaluate(val_generator,
                                                                                steps = (num_train_images//batch_size),
                                                                                #use_multiprocessing=True, #TODO1
                                                                                verbose=0)
                                                        va = {out: va[i] for i, out in enumerate(modelUnet.metrics_names)}
                                                        # move test generator up to where the other ones are if this become permanent
                                                        test_generator = image_segmentation_generator(
                                                            test_frames_path, test_masks_path,  batch_size,  3,
                                                            IMG_SIZE, IMG_SIZE, norm_type)
                                                                                                                
                                                        te = modelUnet.evaluate(test_generator,
                                                                                steps = (num_train_images//batch_size),
                                                                                #use_multiprocessing=True, #TODO1
                                                                                verbose=0)
                                                        te = {out: te[i] for i, out in enumerate(modelUnet.metrics_names)}

                                                        tr = modelUnet.evaluate(train_generator,
                                                                                steps = (num_train_images//batch_size),
                                                                                #use_multiprocessing=True, #TODO1
                                                                                verbose=0)
                                                        tr = {out: tr[i] for i, out in enumerate(modelUnet.metrics_names)}
                                                        print('~~~~~~~~~~')                                                        
                                                        print('Evaluated Validation Accuracy:', va['accuracy'])###
                                                        print('\nHistory Validation Accuracy:', results.history['val_accuracy'][-1])###
                                                        print('\nHistory Training Accuracy:', results.history['accuracy'][-1])###
                                                        print('\nEvaluated Training Accuracy:', tr['accuracy'])
                                                        print('\nTest Accuracy', te['accuracy'])###
                                                        print('~~~~~~~~~~')
                                                        print('Evaluated Validation F1:', va['f1_macro'])###
                                                        print('\nHistory Validation F1:', results.history['val_f1_macro'][-1])###
                                                        print('\nHistory Training F1:', results.history['f1_macro'][-1])###
                                                        print('\nEvaluated Training F1:', tr['f1_macro'])
                                                        print('\nTest F1', te['f1_macro'])###
                                                        print('~~~~~~~~~~')
                                                        
                                                        
                                                        # save the model TODO1 changes to model_name and added model_info
                                                        model_name = str(modelUnet.name) \
                                                            + '_num_' + str(counter) \
                                                            + '_normtype_' + str(norm_type)# make sure norm_type is part of the name, or assess_model() won't work
                                                        model_info = str(modelUnet.name) \
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
                                                            + '_normtype_' + str(norm_type)

                                                        # total number of epochs this model was trained for
                                                        last_epoch = len(results.history[MONITOR]) - 1 # note that the first epoch is "0"
                                                        # number of epochs before early stopping saved the best model
                                                        best_model_epoch = last_epoch - PATIENCE
                                                        # the best F1 score achieved while training this model
                                                        current_f1 = results.history['val_f1_macro'][best_model_epoch]#TODO1 use batch version of val_f1_macro for the compute canada machine
                                                        # if the current model has the best F1 score yet, save it
                                                        if current_f1 > best_f1:
                                                            best_f1 = current_f1
                                                            save_model(modelUnet, results, last_epoch,
                                                                       best_model_epoch, model_name, model_info,#TODO1
                                                                       history_dir, val_history=va, test_history=te)
    print('Finetuning Loop completed')#TODO1 new, for monitoring the script for completion
