from data_loader import image_segmentation_generator
from unet_mini import unet
from save_training import save_model
import os

from keras.callbacks import EarlyStopping
import numpy as np

#TODO
'''
- add proper documentation to this function
- do I actually need to iterate over image sizes?
'''

# hyperparameters
BATCH_SIZE = 16
EPOCHS = 300 # early stopping is bounded by this maximum number of epochs, increase this a lot to run on compute canada
IMG_SIZES = [128]# [256, 512] increase to 1024 to run on compute canada
LR = [1e-2] #[1e-2, 1e-3, 1e-4, 1e-5, 1e-6] use the full set for real finetuning
LOSSES = ['categorical_crossentropy']# replace with these options: ['categorical_crossentropy', 'dice_coef_loss', 'bootstrapped_crossentropy', 'dice_coef_loss_bce']
ACTIVATIONLASTS = ['softmax'] #using only softmax for now, replace with these options afterwards: ['sigmoid','softmax', 'relu']
MAXPOOLINGS = [2,4]
FIRSTFILTERS = [8,16,32,64]
KSIZES = [3,7,15]
DROPS = [True,False]
MORMALIZATIONS = [True,False]
SEED = 10

# early stopping hyperparameters
PATIENCE = 10 # patience for early stopping, train for this many epochs w/ no improvement
MONITOR = 'val_loss' # this metric is monitored to determine stoppage point and best weights
#todo rename optimization to something better
OPTIMIZATION = 'min' # either min or max, depending on MONITOR


def finetuning_loop(history_dir, train_frames_path, train_masks_path, val_frames_path, val_masks_path):
    best_f1 = -1
    counter = 0#cut, counter just tracks how many models have been trained
    for maxpool in MAXPOOLINGS:
        for activationLast in ACTIVATIONLASTS:
            for IMG_SIZE in IMG_SIZES:
                for loss1 in LOSSES:
                    for nfilter1 in FIRSTFILTERS:
                        for lr1 in LR:
                            for kSize1 in KSIZES:
                                for dp in DROPS:
                                    for normali in MORMALIZATIONS:
                                        
                                        # create generators for training and validation images
                                        train_generator = image_segmentation_generator(
                                            train_frames_path, train_masks_path,  BATCH_SIZE,  3,
                                            128, 128)
                                        val_generator = image_segmentation_generator(
                                            val_frames_path, val_masks_path,  BATCH_SIZE,  3,
                                            128, 128)
                                        
                                        NO_OF_TRAINING_IMAGES = len(os.listdir(train_frames_path ))
                                        NO_OF_VAL_IMAGES = len(os.listdir(val_frames_path))
                                        
                                        # build the model

                                        
                                        modelUnet = unet(lr = lr1, input_size = (IMG_SIZE,IMG_SIZE,1), loss_mode = loss1,
                                                         firstFilters = nfilter1, kSize = kSize1,
                                                         activationLast=activationLast, pool_size_max_pooling=maxpool, batchNorm = normali,
                                                         dropOutLayerFlag=dp)
                                        
                                        # print out a summary of the model
                                        #modelUnet.summary() TODO can replace this summary
                                        counter += 1#cut
                                        print('now training the', counter, 'th model')#cut
                                        
                                        # train the model
                                        

                                        # from https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
                                        es = EarlyStopping(monitor=MONITOR, mode=OPTIMIZATION, verbose=1, patience=PATIENCE, restore_best_weights=True)

                                        results = modelUnet.fit_generator(train_generator, epochs=EPOCHS,
                                                                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                                                                          validation_data=val_generator,
                                                                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),verbose=0,
                                                                          callbacks = [es])
                                        
                                        # save the model

                                        
                                        model_name = str(modelUnet.name) + "_IMG_SIZE_" + str(IMG_SIZE) + "_loss1_" + str(
                                            loss1) + "_nfilter1_" + str(
                                                nfilter1) \
                                                + "_lr1_" + str(lr1) + "_kSize1_" + str(kSize1) + "_activationLast_" + str(
                                                    activationLast) \
                                                    + "_maxpool_" + str(maxpool) + "_normali_" + str(normali)\
                                                    + "_dropout_" + str(dp)
                                        print(model_name)
                                        #model_name_full = history_dir + modelName
                                        

                                        # total number of epochs this model was trained for
                                        last_epoch = len(results.history[MONITOR]) - 1 # note that the first epoch is "0"
                                        # number of epochs before early stopping saved the best model
                                        best_model_epoch = last_epoch - PATIENCE
                                        # the best F1 score achieved while training this model
                                        current_f1 = results.history['val_f1_m'][best_model_epoch]

                                        if current_f1 > best_f1:
                                            best_f1 = current_f1
                                            save_model(modelUnet, results, last_epoch, best_model_epoch, model_name, history_dir)###
