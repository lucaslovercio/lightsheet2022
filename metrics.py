from keras import backend as K
#import tensorflow as tf # this is new, might not be needed if we get the confusion matrix from somewhere else
from sklearn.metrics import confusion_matrix # new, for confusion matrix
SMOOTH_LOSS = 1e-12

#TODO
'''
- understand/document the existing functions

- add in Fscore MacroAveraging
 - should it be batchwise or epoch wise using a callback?
 - is it not already good for macro averaging?

- accuracy of tissue vs. not tissue
- need to retrieve the confusion matrix, and not the y_true and y_pred (disposition of true and predicted labels, rows?, columns?

ideas:
keras has a metric called confusion_matrix (how can this be visualized in matplotlib?)

custom metrics:
need to take y_true and y_pred as arguments
need to return a single tensor value
the returned tensors have shape (# of rows, 1)
'''


# below here are from https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/unet/metrics.py
def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)
    return K.mean(jac)

def jacard_coef_flat(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + SMOOTH_LOSS) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + SMOOTH_LOSS)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


#below here are from Lucas
#TODO does the m stand for? (ask lucas)
#TODO insert < y_true = K.ones_like(y_true) > ?
#TODO add beta to the f1 score?
#TODO move the sums from in the recall and precision functions to in the f1 function
def recall_m(y_true, y_pred):
    y_true
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#below here are new
#TODO get a working confusion matrix for inter_tissue_accuracy
def inter_tissue_accuracy(y_true, y_pred):
    #confusion = tf.math.confusion_matrix(y_true, y_pred)
    # print("shape###################################", y_true.shape)###
    # print("shape###################################", y_pred.shape)###
    #confusion = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))###
    #print("confusion:", confusion)###
    
    return f1_m(y_true, y_pred)#TODO cut this!!!
    #TODO return a tensor of the accuracy

#TODO see if the below actually make any sense
def recall_M(y_true, y_pred):
    y_true
    true_positives = K.round(K.clip(y_true * y_pred, 0, 1))
    possible_positives = K.round(K.clip(y_true, 0, 1))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_M(y_true, y_pred):
    true_positives = K.round(K.clip(y_true * y_pred, 0, 1))
    predicted_positives = K.round(K.clip(y_pred, 0, 1))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_M(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return K.sum(2*((precision*recall)/(precision+recall+K.epsilon())))

