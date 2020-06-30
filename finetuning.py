from data_loader import image_segmentation_generator
from unet_mini import unet
from save_training import showLosses,saveLossesTXT, saveModelSummary
import os

from keras.callbacks import EarlyStopping#new
from keras.callbacks import ModelCheckpoint#new
#TODO
'''
- fix up the modelcheckpoint/earlystopping functionality so that they work with save_training
- add proper documentation to this function
'''

# global variables
BATCH_SIZE = 16
EPOCHS = 30
IMG_SIZES = [128]# [256, 512]
LR = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
LOSSES = ['categorical_crossentropy',] #['bcedice', 'binary_crossentropy']
ACTIVATIONLASTS = ['softmax'] #using only softmax for now, replace with these options afterwards ['sigmoid','softmax', 'relu']
MAXPOOLINGS = [2,4]
FIRSTFILTERS = [8,16,32,64]
KSIZES = [3,7,15]
DROPS = [True,False]
MORMALIZATIONS = [True,False]
SEED = 10

train_frames_path   = 'TissueDataset/Training/Original'
train_masks_path    = 'TissueDataset/Training/Mask'

val_frames_path     = 'TissueDataset/Validation/Mask'
val_masks_path      = 'TissueDataset/Validation/Mask'

def finetuning_loop(history_dir, train_frames_path, train_masks_path, val_frames_path, val_masks_path):
    bestF1 = -1
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
                                        modelUnet.summary()
                                        
                                        # train the model
                                        #new starts here
                                        # from https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
                                        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
                                        mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
                                        #new ends here
                                        results = modelUnet.fit_generator(train_generator, epochs=EPOCHS,
                                                                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                                                                          validation_data=val_generator,
                                                                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),verbose=0,
                                                                          callbacks = [es, mc])#TODO new
                                        
                                        # save the model
                                        modelName = str(modelUnet.name) + "_IMG_SIZE_" + str(IMG_SIZE) + "_loss1_" + str(
                                            loss1) + "_nfilter1_" + str(
                                                nfilter1) \
                                                + "_lr1_" + str(lr1) + "_kSize1_" + str(kSize1) + "_activationLast_" + str(
                                                    activationLast) \
                                                    + "_maxpool_" + str(maxpool) + "_normali_" + str(normali)\
                                                    + "_dropout_" + str(dp)
                                        print(modelName)
                                        modelNameFull = history_dir + modelName
                                        
                                        showLosses(results, EPOCHS, modelNameFull)
                                        
                                        vF1 = results.history['val_f1_m']
                                        if vF1[-1] > bestF1:
                                            bestF1 = vF1[-1]
                                            saveLossesTXT(results, modelNameFull)
                                            modelFileFull = modelNameFull + ".h5"
                                            modelUnet.save(modelFileFull)
                                            saveModelSummary(modelUnet, modelNameFull)
