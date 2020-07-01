import tensorflow as tf
from keras.backend.tensorflow_backend import _to_tensor
#TODO
'''
'''

#copied from https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/unet/losses.py

import metrics

def dice_coef_loss(y_true, y_pred):
    return 1 - metrics.dice_coef(y_true, y_pred)


def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
    target_tensor = y_true
    prediction_tensor = y_pred

    _epsilon = _to_tensor(tf.keras.backend.epsilon(), prediction_tensor.dtype.base_dtype)
    prediction_tensor = tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
    prediction_tensor = tf.keras.backend.log(prediction_tensor / (1 - prediction_tensor))

    if bootstrap_type == 'soft':
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * tf.sigmoid(prediction_tensor)
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * tf.cast(
            tf.sigmoid(prediction_tensor) > 0.5, tf.float32)
    return tf.keras.backend.mean(tf.nn.sigmoid_cross_entropy_with_logits(#TODO not sure if this should be treating prediction_tensor as logits
        labels=bootstrap_target_tensor, logits=prediction_tensor))

def dice_coef_loss_bce(y_true, y_pred):
    dice = 0.8
    bce = 0.2
    bootstrapping = 'hard'
    alpha = 1.
    return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice
