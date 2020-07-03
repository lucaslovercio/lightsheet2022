from keras.models import *
from keras.layers import *
from keras.optimizers import *
import losses
import metrics
#TODO
'''
- clean up
- consider adding the activation between layers as a finetunable hyperparam (at least try out leaky relu, softplus, etc.)
- update model call to keras 2 API?
'''

#TODO extend this to a wider array of possibilities
def get_loss_func(loss_mode):

    if loss_mode == "bcedice":
        loss_func = losses.dice_coef_loss_bce
    elif loss_mode == "categorical_crossentropy":
        loss_func = loss_mode
    else:
        loss_func = "binary_crossentropy"
    #print("loss:", loss_func)###
    
    return loss_func

def unet(lr=1e-4, input_size=(256, 256, 1), loss_mode='binary_crossentropy', firstFilters=32, kSize=3,
         pool_size_max_pooling=2, activationLast='softmax', batchNorm = True,
         dropOutLayerFlag = True, dropOutLayerRatio = 0.3, nClasses = 3):

    concat_axis = 3
    activation = 'relu'
    inputs = Input(input_size)

    #conv1a
    conv1 = Conv2D(firstFilters, (kSize, kSize), padding='same')(inputs)
    if batchNorm:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)

    #conv1b
    conv1 = Conv2D(firstFilters, (kSize, kSize), padding='same')(inputs)
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

    o = Conv2D(nClasses, (1, 1), padding='same',activation=activationLast)(conv5)

    model = Model(input=inputs, output=o)
    model.name = "unetMini4"

    loss_func = get_loss_func(loss_mode)
    model.compile(optimizer=Adam(lr=lr), loss=loss_func, metrics=[metrics.jaccard_coef, metrics.jacard_coef_flat,
                                                                   metrics.jaccard_coef_int, metrics.dice_coef,
                                                                   metrics.recall_m, metrics.precision_m, metrics.f1_m,
                                                                   #metrics.inter_tissue_accuracy, #this is new, and doesn't work yet
                                                                   metrics.background_accuracy, #TODO newish
                                                                   #TODO1 below here are new metrics
                                                                   metrics.f1_M,
                                                                   metrics.recall_M,
                                                                   metrics.precision_M,
                                                                   metrics.recall_0,
                                                                   metrics.precision_0,
                                                                   metrics.recall_1,
                                                                   metrics.precision_1,
                                                                   metrics.recall_2,
                                                                   metrics.precision_2, 
                                                                   'accuracy'])
    return model
