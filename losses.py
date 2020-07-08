import tensorflow as tf
import keras.backend as K

#TODO
'''
'''


#from https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

    return 1 - (numerator / (denominator + K.epsilon()))


def dice_crossentropy_loss(dice_weight=0.5):
    crossentropy_weight = 1.0 - dice_weight
    cce = tf.keras.losses.CategoricalCrossentropy()
    def loss(y_true, y_pred):
        return  dice_weight * dice_loss(y_true, y_pred) + crossentropy_weight * cce(y_true, y_pred)
    return loss
