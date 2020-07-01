from keras import backend as K
import tensorflow as tf
#from sklearn.metrics import confusion_matrix # new, for confusion matrix
SMOOTH_LOSS = 1e-12 #TODO should maybe replace smooth with a call to K.epsilon()?

#TODO
'''
- does existing Fscore MacroAveraging work?
'''

'''
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


def dice_coef(y_true, y_pred, smooth=1.0):#used in losses
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


#below here are from Lucas
#TODO does the m stand for? (ask lucas)
#TODO add beta to the f1 score?
#TODO move the sums from in the recall and precision functions to in the f1 function
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred, beta=1.0):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall + K.epsilon()))

#TODO deprecated, ask if this is at all useful
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
    
    #return y_true TODO1 original
    return correct_count / total_count


def background_accuracy(y_true, y_pred):
    background_true = y_true[:,:,:,0]#cut away the tissue
    background_pred = y_pred[:,:,:,0]#cut away the tissue
    
    #TODO1 if the output isn't probabilistic we don't need the line below
    background_pred = K.round(K.clip(background_pred, 0, 1))
    
    
    boolean_tensor = tf.math.equal(background_true, background_pred)
    correct_count = tf.dtypes.cast(boolean_tensor, tf.int32)
    correct_count = tf.math.reduce_sum(correct_count)
    
    total_count = tf.size(background_true)
    
    return correct_count / total_count


#TODO this function is deprecated
def f1_M(y_true, y_pred, beta=1):
    # TODO account for these strange shapes
    # it seems y_true has shape (None, None, None, None) in training is really (16, 128, 128, 3)
    # it seems y_pred has shape (None, 128, 128, 3)
    # tensorflow uses None to mark a shape it doesn't know, but how doesn't it know?
    # current theory is that converting to numpy arrays is impossible, and the original implementation does the right thing
    # https://stackoverflow.com/questions/51100508/implementing-custom-loss-function-in-keras-with-condition 
    # could add beta to original version

    #TODO1 below is basically a thought experiment
    # compress y_true and y_pred from having there values encoded in three channels at the end to a flat batch_size x 128 x 128 matrix
    y_true = y_true.argmax(axis=-1) # new
    y_pred = y_pred.argmax(axis=-1) # new
    #numpy .multiply is element wise
    tp = sum(np.multiply(y_true, y_pred))
    fp = sum(y_pred) - tp
    tn = sum(np.multiply(y_true ^ 1, y_pred ^ 1))
    fn = sum(y_pred ^ 1) - tn

    num_classes = 3#leave this fixed for now
    # TODO might need to convert tp, fp, tn, fn to floats before doing division with them
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    return (1 + beta**2) * (precision * recall) / (beta * precision + recall)
