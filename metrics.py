from keras import backend as K
import tensorflow as tf
SMOOTH_LOSS = 1e-12 #TODO should maybe replace smooth with a call to K.epsilon()?

#TODO
'''
- double check and understand Olga's jaccard and dice metrics
- look into combining f1_m and f1_M into a function that's passed a flag 'macro' or 'micro'
- if needed, implement epoch-wise precision and recall metrics (as stateful custom metrics)
- if needed, implement epoch-wise accuracy etc. metrics (as stateful custom metrics)
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

# end of recall and precision for specific classes

# accuracy b/w the two types of tissues
def tissue_type_accuracy(y_true, y_pred):

    ###
    background_mask = 1.0 - y_true[:,:,:,0]
    tissue_true = tf.math.argmax(y_true[:,:,:,1:], axis=-1)
    #tissue_true = tf.boolean_mask(tissue_true, background_mask)
    tissue_pred = tf.math.argmax(y_pred[:,:,:,1:], axis=-1)
    #tissue_pred = tf.boolean_mask(tissue_pred, background_mask)

    true_positive_mat = tf.boolean_mask(tf.cast(tf.equal(tissue_true, tissue_pred), tf.float32), background_mask)
    true_positives = tf.math.reduce_sum(true_positive_mat)
    total_predictions = tf.cast(tf.size(true_positive_mat), tf.float32)
    return true_positives / total_predictions
    #true_positives = tf.math.reduce_sum(tf.cast(tf.equal(tissue_true, tissue_pred), dtype=float_32))
    # then divide true positives by the size of tissue_true (or tissue pred

# accuracy of tissue vs. background segmentations
def binary_accuracy(y_true, y_pred):
    #cut away the tissue
    background_true = y_true[:,:,:,0]
    background_pred = y_pred[:,:,:,0]
    #turn probabilities into predictions
    background_pred = K.round(K.clip(background_pred, 0, 1))
    
    #TODO re-examine this and try to replace tf.math. w/ K.
    boolean_tensor = tf.math.equal(background_true, background_pred)
    correct_count = tf.dtypes.cast(boolean_tensor, tf.int32)
    correct_count = tf.math.reduce_sum(correct_count)
    
    total_count = tf.size(background_true)
    
    return correct_count / total_count


########### below here are stateful versions of metrics

# tensorflow documentaion https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric
# other documentation https://neptune.ai/blog/keras-metrics

        
#TODO add beta?        
class F1Micro(tf.keras.metrics.Metric):
    def __init__(self, name='f1_micro', **kwargs):
        super(F1Micro, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')#TODO what's the point of name='tp', etc. here
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred):
        max_probabilities = tf.math.reduce_max(y_pred, axis=-1, keepdims=True)
        y_pred = tf.math.equal(y_pred, max_probabilities)#note this should broadcast the final dim of max_probabilities
        # convert the one hot ground truth to boolean
        y_true = tf.cast(y_true, 'int32') == 1
        correct_pred = tf.math.equal(y_pred, y_true)
        tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true, correct_pred), 'float32'))
        tn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true), correct_pred), 'float32'))
        fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true), tf.math.logical_not(correct_pred)), 'float32'))
        fn = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true, tf.math.logical_not(correct_pred)), 'float32'))
        self.true_positives.assign_add(tp)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
        
        
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        return 2 * (precision * recall) / (precision + recall)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)
        self.true_negatives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

#TODO right now this method relies on there being exactly 3 classes, should be made to generalize better probably
class F1Macro(tf.keras.metrics.Metric):
    def __init__(self, name='f1_macro', **kwargs):
        super(F1Macro, self).__init__(name=name, **kwargs)
        self.true_positives_0 = self.add_weight(name='tp0', initializer='zeros')
        self.false_positives_0 = self.add_weight(name='fp0', initializer='zeros')
        self.false_negatives_0 = self.add_weight(name='fn0', initializer='zeros')
        #self.recall_0 = self.add_weight(name='recall_0', initializer='zeros')
        #self.precision_0 = self.add_weight(name='precision_0', initializer='zeros')

        self.true_positives_1 = self.add_weight(name='tp1', initializer='zeros')
        self.false_positives_1 = self.add_weight(name='fp1', initializer='zeros')
        self.false_negatives_1 = self.add_weight(name='fn1', initializer='zeros')
        #self.recall_1 = self.add_weight(name='recall_1', initializer='zeros')
        #self.precision_1 = self.add_weight(name='precision_1', initializer='zeros')

        self.true_positives_2 = self.add_weight(name='tp2', initializer='zeros')
        self.false_positives_2 = self.add_weight(name='fp2', initializer='zeros')
        self.false_negatives_2 = self.add_weight(name='fn2', initializer='zeros')
        #self.recall_2 = self.add_weight(name='recall_2', initializer='zeros')
        #self.precision_2 = self.add_weight(name='precision_2', initializer='zeros')

    def update_state(self, y_true, y_pred):
        max_probabilities = tf.math.reduce_max(y_pred, axis=-1, keepdims=True)
        y_pred = tf.math.equal(y_pred, max_probabilities)#note this should broadcast the final dim of max_probabilities
        # convert the one hot ground truth to boolean
        y_true = tf.cast(y_true, 'int32') == 1
        correct_pred = tf.math.equal(y_pred, y_true)
        tp0 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true[:,:,:,0], correct_pred[:,:,:,0]), 'float32'))
        fp0 = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true[:,:,:,0]), tf.math.logical_not(correct_pred[:,:,:,0])), 'float32'))
        fn0 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true[:,:,:,0], tf.math.logical_not(correct_pred[:,:,:,0])), 'float32'))
        
        tp1 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true[:,:,:,1], correct_pred[:,:,:,1]), 'float32'))
        fp1 = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true[:,:,:,1]), tf.math.logical_not(correct_pred[:,:,:,1])), 'float32'))
        fn1 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true[:,:,:,1], tf.math.logical_not(correct_pred[:,:,:,1])), 'float32'))
        
        tp2 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true[:,:,:,2], correct_pred[:,:,:,2]), 'float32'))
        fp2 = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true[:,:,:,2]), tf.math.logical_not(correct_pred[:,:,:,2])), 'float32'))
        fn2 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true[:,:,:,2], tf.math.logical_not(correct_pred[:,:,:,2])), 'float32'))
        
        self.true_positives_0.assign_add(tp0)
        self.false_positives_0.assign_add(fp0)
        self.false_negatives_0.assign_add(fn0)

        self.true_positives_1.assign_add(tp1)
        self.false_positives_1.assign_add(fp1)
        self.false_negatives_1.assign_add(fn1)
        
        self.true_positives_2.assign_add(tp2)
        self.false_positives_2.assign_add(fp2)
        self.false_negatives_2.assign_add(fn2)
        
        
    def result(self):
        precision_0 = self.true_positives_0 / (self.true_positives_0 + self.false_positives_0)
        recall_0 = self.true_positives_0 / (self.true_positives_0 + self.false_negatives_0)

        precision_1 = self.true_positives_1 / (self.true_positives_1 + self.false_positives_1)
        recall_1 = self.true_positives_1 / (self.true_positives_1 + self.false_negatives_1)

        precision_2 = self.true_positives_2 / (self.true_positives_2 + self.false_positives_2)
        recall_2 = self.true_positives_2 / (self.true_positives_2 + self.false_negatives_2)

        average_precision = (precision_0 + precision_1 + precision_2) / 3.0
        average_recall = (recall_0 + recall_1 + recall_2) / 3.0

        return 2 * (average_precision * average_recall) / (average_precision + average_recall)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)
        self.true_negatives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)
