from keras import backend as K
import tensorflow as tf
SMOOTH_LOSS = 1e-12 #TODO should maybe replace smooth with a call to K.epsilon()?

#TODO
'''
- rename background accuracy to something better
- delete deprecated metrics
- double check and understand Olga's jaccard and dice metrics
- look into combining f1_m and f1_M into a function that's passed a flag 'macro' or 'micro'
'''

'''
custom metrics:
need to take y_true and y_pred as arguments
need to return a single tensor value
the returned tensors have shape (# of rows, 1)

y_pred:
The output of the last model layer, softmax probabilities in a vector of size <# of classes> for the current model.
Has shape = (batch_size, 128, 128, 3).

y_true:
The ground truth segmentations, 1-hot encoded.
Has shape = (batch_size, 128, 128, 3).
'''


########### below here are from https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/unet/metrics.py
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


def dice_coef(y_true, y_pred, smooth=1.0):#used in losses
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# microaveraged recall
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# microaveraged precision
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# microaveraged f1
def f1_m(y_true, y_pred, beta=1.0):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall + K.epsilon()))

########### below here are new

# macroaveraged recall
def recall_M(y_true, y_pred):
    recall = 0
    num_classes = y_pred.shape[-1]
    for c in range(num_classes):
        recall += recall_m(y_true[:,:,:,c], y_pred[:,:,:,c])
    return recall / num_classes



# macroaveraged precision
def precision_M(y_true, y_pred):
    precision = 0
    num_classes = y_pred.shape[-1]
    for c in range(num_classes):
        precision += precision_m(y_true[:,:,:,c], y_pred[:,:,:,c])
    return precision / num_classes



#macroaveraged f1
def f1_M(y_true, y_pred, beta=1.0):
    precision = precision_M(y_true, y_pred)
    recall = recall_M(y_true, y_pred)
    return (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall + K.epsilon()))

# recall and precision for specific classes
def recall_0(y_true, y_pred):
    return recall_m(y_true[:,:,:,0], y_pred[:,:,:,0])

def precision_0(y_true, y_pred):
    return precision_m(y_true[:,:,:,0], y_pred[:,:,:,0])


def recall_1(y_true, y_pred):
    return recall_m(y_true[:,:,:,1], y_pred[:,:,:,1])

def precision_1(y_true, y_pred):
    return precision_m(y_true[:,:,:,1], y_pred[:,:,:,1])


def recall_2(y_true, y_pred):
    return recall_m(y_true[:,:,:,2], y_pred[:,:,:,2])

def precision_2(y_true, y_pred):
    return precision_m(y_true[:,:,:,2], y_pred[:,:,:,2])







#TODO fix this up, want accuracy b/w the two types of tissues
def inter_tissue_accuracy(y_true, y_pred):
    #confusion = tf.math.confusion_matrix(y_true, y_pred)
    #confusion = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))###
    #print("confusion:", confusion)###
    #y_true = K.ones_like(y_true)?
    #print("shape###################################", y_pred.shape, y_pred)###

    tissue_true = y_true[:,:,:,1:]#cut away the background class
    tissue_pred = y_pred[:,:,:,1:]#cut away the background class
    
    #TODO if the output isn't probabilistic we don't need the line below
    tissue_pred = K.round(K.clip(tissue_pred, 0, 1))
    

    boolean_tensor = tf.math.equal(tissue_true, tissue_pred)
    correct_count = tf.dtypes.cast(boolean_tensor, tf.int32)
    correct_count = tf.math.reduce_sum(correct_count)

    total_count = tf.size(tissue_pred)
    #print(total_count)###

    #seems that this division automatically converts to float64
    #print(correct_count / total_count)###
    
    #return y_true 
    return correct_count / total_count

#TODO rename this
def background_accuracy(y_true, y_pred):
    background_true = y_true[:,:,:,0]#cut away the tissue
    background_pred = y_pred[:,:,:,0]#cut away the tissue
    background_pred = K.round(K.clip(background_pred, 0, 1))#turn probabilities into predictions
    
    #TODO re-examine this and try to replace tf.math. w/ K.
    boolean_tensor = tf.math.equal(background_true, background_pred)
    correct_count = tf.dtypes.cast(boolean_tensor, tf.int32)
    correct_count = tf.math.reduce_sum(correct_count)
    
    total_count = tf.size(background_true)
    
    return correct_count / total_count
