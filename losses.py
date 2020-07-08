import tensorflow as tf
import keras.backend as K

#TODO
'''
'''


#from https://www.jeremyjordan.me/semantic-segmentation/#loss
# also reference https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
# really is soft dice since argmax has no derivative and can't be used for backprop
def dice(y_true, y_pred):
    # the numerator approximates 2 * tp, is a vector of length <number of classes>
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred)
    # denominator approximates 2 * tp + fn + fp, is a vector of length <number of classes>
    denominator = tf.reduce_sum(y_true + y_pred)#TODO should these be squared?
    # return the (macro)average f1 score (approximation) for all classes, negated for minimization purposes
    return 1 - tf.reduce_mean(numerator / denominator)


def dice_cce(dice_weight=0.5):
    crossentropy_weight = 1.0 - dice_weight
    cce = tf.keras.losses.CategoricalCrossentropy()
    def loss(y_true, y_pred):
        return  dice_weight * dice(y_true, y_pred) + crossentropy_weight * cce(y_true, y_pred)
    return loss
