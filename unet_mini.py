from keras.models import *
from keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop

import losses
import metrics
#TODO
'''
- clean up
'''

def get_loss_func(loss_mode):

    if loss_mode == 'dice0_cce100':
        return 'categorical_crossentropy'
    elif loss_mode == 'dice20_cce80':
        return losses.dice_cce(0.2)
    elif loss_mode == 'dice50_cce50':
        return losses.dice_cce(0.5)
    elif loss_mode == 'dice100_cce0':
        return losses.dice
    elif loss_mode == 'dice':
        return losses.dice
    elif loss_mode == 'categorical_crossentropy':
        return 'categorical_crossentropy'
    else:
        print('Loss function "'  + loss_mode + '" is not defined, so training with categorical_crossentropy instead')
    return 'categorical_crossentropy'

def get_optimizer(opt, lr):
    if opt == 'rmsprop':
        return RMSprop(lr)
    else:
        return Adam(lr)
        

def unet(lr=1e-4, input_size=(256, 256, 1), loss_mode='categorical_crossentropy', firstFilters=32, kSize=3,
         pool_size_max_pooling=2, activation_last='softmax', batchNorm = True,
         dropOutLayerFlag = True, dropOutLayerRatio = 0.3, nClasses = 3, activation = 'relu',
         optimizer='adam'):

    concat_axis = 3
    inputs = Input(input_size)

    #conv1a
    conv1 = Conv2D(firstFilters, (kSize, kSize), padding='same')(inputs)
    if batchNorm:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)

    #conv1b
    conv1 = Conv2D(firstFilters, (kSize, kSize), padding='same')(conv1)
    if batchNorm:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)

    pool1 = MaxPooling2D((pool_size_max_pooling, pool_size_max_pooling))(conv1)

    # conv2a
    conv2 = Conv2D(firstFilters, (kSize, kSize), padding='same')(pool1)
    if batchNorm:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)

    # conv2b
    conv2 = Conv2D(firstFilters, (kSize, kSize), padding='same')(conv2)
    if batchNorm:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)

    #Before the pooling layer, as in https://github.com/naomifridman/Unet_Brain_tumor_segmentation/blob/master/model_unet.py
    if dropOutLayerFlag:
        conv2 = Dropout(dropOutLayerRatio)(conv2)

    pool2 = MaxPooling2D((pool_size_max_pooling, pool_size_max_pooling))(conv2)

    # conv3a
    conv3 = Conv2D(firstFilters, (kSize, kSize), padding='same')(pool2)
    if batchNorm:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)

    # conv3b
    conv3 = Conv2D(firstFilters, (kSize, kSize), padding='same')(conv3)
    if batchNorm:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)

    # According to Ronneberger 2015 . Drop-out layers at the end of the contracting path perform further implicit data augmentation
    if dropOutLayerFlag:
        conv3 = Dropout(dropOutLayerRatio)(conv3)

    # up1
    up1 = concatenate([UpSampling2D((pool_size_max_pooling, pool_size_max_pooling))(
        conv3), conv2], axis=concat_axis)

    # conv4a
    conv4 = Conv2D(firstFilters, (kSize, kSize), padding='same')(up1)
    if batchNorm:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)

    # conv4b
    conv4 = Conv2D(firstFilters, (kSize, kSize), padding='same')(conv4)
    if batchNorm:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)

    # up2
    up2 = concatenate([UpSampling2D((pool_size_max_pooling, pool_size_max_pooling))(
        conv4), conv1], axis=concat_axis)

    # conv5a
    conv5 = Conv2D(firstFilters, (kSize, kSize), padding='same')(up2)
    if batchNorm:
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)

    # conv5b
    conv5 = Conv2D(firstFilters, (kSize, kSize), padding='same')(conv5)
    if batchNorm:
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)

    o = Conv2D(nClasses, (1, 1), padding='same',activation=activation_last)(conv5)

    model = Model(inputs=inputs, outputs=o, name = "unetMini")
    #model.name = "unetMini"

    loss_func = get_loss_func(loss_mode)
    optimizer = get_optimizer(optimizer, lr)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=[# batch-averaged precision recall and f1
                                                                metrics.recall_macro_batch,
                                                                metrics.precision_macro_batch,
                                                                metrics.f1_macro_batch,
                                                                # batch-averaged precision and recall for each class
                                                                metrics.reca_0,
                                                                metrics.prec_0,
                                                                metrics.reca_1,
                                                                metrics.prec_1,
                                                                metrics.reca_2,
                                                                metrics.prec_2,
                                                                # epochwise metrics
                                                                metrics.RecallMacro(), 
                                                                metrics.PrecisionMacro(),
                                                                metrics.F1Macro(),
                                                                metrics.RecallMacro0(),
                                                                metrics.PrecisionMacro0(),
                                                                metrics.F1Macro0(),
                                                                metrics.RecallMacro1(),
                                                                metrics.PrecisionMacro1(),
                                                                metrics.F1Macro1(),
                                                                metrics.RecallMacro2(),
                                                                metrics.PrecisionMacro2(),
                                                                metrics.F1Macro2(),
                                                                # accuracy metrics
                                                                metrics.binary_accuracy_batch,
                                                                metrics.tissue_type_accuracy_batch, 
                                                                metrics.BinaryAccuracy(),
                                                                metrics.TissueTypeAccuracy(), 
                                                                'accuracy'])
    return model

def unet4levels(lr=1e-4, input_size=(256, 256, 1), loss_mode='categorical_crossentropy', firstFilters=32, kSize=3,
         pool_size_max_pooling=2, activation_last='softmax', batchNorm = True,
         dropOutLayerFlag = True, dropOutLayerRatio = 0.3, nClasses = 3, activation = 'relu',
         optimizer='adam'):

    concat_axis = 3
    inputs = Input(input_size)

    #conv1a
    conv1 = Conv2D(firstFilters, (kSize, kSize), padding='same')(inputs)
    if batchNorm:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)

    #conv1b
    conv1 = Conv2D(firstFilters, (kSize, kSize), padding='same')(conv1)
    if batchNorm:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)

    pool1 = MaxPooling2D((pool_size_max_pooling, pool_size_max_pooling))(conv1)

    # conv2a
    conv2 = Conv2D(firstFilters * 2, (kSize, kSize), padding='same')(pool1)
    if batchNorm:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)

    # conv2b
    conv2 = Conv2D(firstFilters * 2, (kSize, kSize), padding='same')(conv2)
    if batchNorm:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)

    #Before the pooling layer, as in https://github.com/naomifridman/Unet_Brain_tumor_segmentation/blob/master/model_unet.py
    if dropOutLayerFlag:
        conv2 = Dropout(dropOutLayerRatio)(conv2)

    pool2 = MaxPooling2D((pool_size_max_pooling, pool_size_max_pooling))(conv2)

    # conv2a_added
    conv2_added = Conv2D(firstFilters * 4, (kSize, kSize), padding='same')(pool2)
    if batchNorm:
        conv2_added = BatchNormalization()(conv2_added)
    conv2_added = Activation(activation)(conv2_added)

    # conv2b_added
    conv2_added = Conv2D(firstFilters * 4, (kSize, kSize), padding='same')(conv2_added)
    if batchNorm:
        conv2_added = BatchNormalization()(conv2_added)
    conv2_added = Activation(activation)(conv2_added)

    if dropOutLayerFlag:
        conv2_added = Dropout(dropOutLayerRatio)(conv2_added)

    pool2_added = MaxPooling2D((pool_size_max_pooling, pool_size_max_pooling))(conv2_added)

    # conv3a
    conv3 = Conv2D(firstFilters * 8, (kSize, kSize), padding='same')(pool2_added)
    if batchNorm:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)

    # conv3b
    conv3 = Conv2D(firstFilters * 8, (kSize, kSize), padding='same')(conv3)
    if batchNorm:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)

    # According to Ronneberger 2015 . Drop-out layers at the end of the contracting path perform further implicit data augmentation
    if dropOutLayerFlag:
        conv3 = Dropout(dropOutLayerRatio)(conv3)

    # up1_added
    up1_added = concatenate([UpSampling2D((pool_size_max_pooling, pool_size_max_pooling))(
        conv3), conv2_added], axis=concat_axis)

    # conv4_added a
    conv4_added = Conv2D(firstFilters * 4, (kSize, kSize), padding='same')(up1_added)
    if batchNorm:
        conv4_added = BatchNormalization()(conv4_added)
    conv4_added = Activation(activation)(conv4_added)

    # conv4_added b
    conv4_added = Conv2D(firstFilters * 4, (kSize, kSize), padding='same')(conv4_added)
    if batchNorm:
        conv4_added = BatchNormalization()(conv4_added)
    conv4_added = Activation(activation)(conv4_added)

    # up1
    up1 = concatenate([UpSampling2D((pool_size_max_pooling, pool_size_max_pooling))(
        conv4_added), conv2], axis=concat_axis)

    # conv4a
    conv4 = Conv2D(firstFilters * 2, (kSize, kSize), padding='same')(up1)
    if batchNorm:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)

    # conv4b
    conv4 = Conv2D(firstFilters * 2, (kSize, kSize), padding='same')(conv4)
    if batchNorm:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)

    # up2
    up2 = concatenate([UpSampling2D((pool_size_max_pooling, pool_size_max_pooling))(
        conv4), conv1], axis=concat_axis)

    # conv5a
    conv5 = Conv2D(firstFilters, (kSize, kSize), padding='same')(up2)
    if batchNorm:
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)

    # conv5b
    conv5 = Conv2D(firstFilters, (kSize, kSize), padding='same')(conv5)
    if batchNorm:
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)

    o = Conv2D(nClasses, (1, 1), padding='same',activation=activation_last)(conv5)

    model = Model(inputs=inputs, outputs=o, name = "unet4levels")
    #model.name = "unetMini"

    loss_func = get_loss_func(loss_mode)
    optimizer = get_optimizer(optimizer, lr)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=[# batch-averaged precision recall and f1
                                                                metrics.recall_macro_batch,
                                                                metrics.precision_macro_batch,
                                                                metrics.f1_macro_batch,
                                                                # batch-averaged precision and recall for each class
                                                                metrics.reca_0,
                                                                metrics.prec_0,
                                                                metrics.reca_1,
                                                                metrics.prec_1,
                                                                metrics.reca_2,
                                                                metrics.prec_2,
                                                                # epochwise metrics
                                                                metrics.RecallMacro(),
                                                                metrics.PrecisionMacro(),
                                                                metrics.F1Macro(),
                                                                metrics.RecallMacro0(),
                                                                metrics.PrecisionMacro0(),
                                                                metrics.F1Macro0(),
                                                                metrics.RecallMacro1(),
                                                                metrics.PrecisionMacro1(),
                                                                metrics.F1Macro1(),
                                                                metrics.RecallMacro2(),
                                                                metrics.PrecisionMacro2(),
                                                                metrics.F1Macro2(),
                                                                # accuracy metrics
                                                                metrics.binary_accuracy_batch,
                                                                metrics.tissue_type_accuracy_batch,
                                                                metrics.BinaryAccuracy(),
                                                                metrics.TissueTypeAccuracy(),
                                                                'accuracy'])
    return model
