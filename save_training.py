import matplotlib.pyplot as plt
# the commented lines below are a possible solution to thread issues caused by tkinter
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
import os
import numpy as np
import skimage.io as io
#import cv2#TODO cut
#TODO
'''
- need to add some documentation
- consider changing what color and linetypes are used to make the learning curves
- add a more thorough set of metrics to the saved text file
- add a boolean flag "unique" to plot learning curves, which determines whether or not the entire model name is in the saved image names
'''

'''
Name: plot_learning_curves
Purpose:
This function creates and saves plots of model performance metrics.
Arguments:
- history:
 The training history Keras saves during the call to model.fit.
- last_epoch:
 The total number of epochs the model was run for - 1 (start with epoch 0).
- best_model_epoch:
 The epoch at which the model's weights was done training. Early stopping caused several 
 epochs to run after this point, which is marked with a dot on the plots.
- model_name:
 The full filepath to and name of the model.
Outputs:
Saves plots to the directory with the model weights.
'''
def plot_learning_curves(history, last_epoch, best_model_epoch, model_name):

    # plot training and validation accuracy values
    plt.plot(history.history['accuracy'], 'C0-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_accuracy'], 'C0--o', markevery=[best_model_epoch])
    plt.plot(history.history['binary_accuracy'], 'C1-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_binary_accuracy'], 'C1--o', markevery=[best_model_epoch])
    plt.plot(history.history['tissue_type_accuracy'], 'C2-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_tissue_type_accuracy'], 'C2--o', markevery=[best_model_epoch])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Acc', 'Val Acc', 'Train Binary Acc', 'Val Binary Acc', 'Train Tissue Acc', 'Val Tissue Acc'], loc='upper left')
    plt.xlim([0,last_epoch])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_acc.png')
    plt.close()

    # plot training and validation loss values
    plt.plot(history.history['loss'], marker='o', markevery=[best_model_epoch])
    plt.plot(history.history['val_loss'], marker='o', markevery=[best_model_epoch])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Val Loss'], loc='lower left')
    plt.xlim([0,last_epoch])
    plt.ylim(0, max(history.history['loss']) + 1)
    #plt.show()
    plt.savefig(model_name + '_losses.png')
    plt.close()

    # plot training and validation f1 scores
    plt.plot(history.history['f1_m'], 'b-|', markevery=[best_model_epoch])
    plt.plot(history.history['f1_M'], 'b--|', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_m'], 'r-|', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_M'], 'r--|', markevery=[best_model_epoch])
    plt.title('Model F1 Metrics')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Train F1 Micro', 'Train F1 Macro', 'Val F1 Micro', 'Val F1 Macro'], loc='lower left')
    plt.xlim([0, last_epoch])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_f1.png')
    plt.close()

    # plot in depth recall and precision for validation and training TODO should cut this graph
    plt.plot(history.history['recall_m'], 'b-|', markevery=[best_model_epoch])
    plt.plot(history.history['precision_m'], 'b--|', markevery=[best_model_epoch])
    plt.plot(history.history['recall_M'], 'b:|', markevery=[best_model_epoch])
    plt.plot(history.history['precision_M'], 'b-.|', markevery=[best_model_epoch])    
    plt.plot(history.history['val_recall_m'], 'r-|', markevery=[best_model_epoch])
    plt.plot(history.history['val_precision_m'], 'r--|', markevery=[best_model_epoch])
    plt.plot(history.history['val_recall_M'], 'r:|', markevery=[best_model_epoch])
    plt.plot(history.history['val_precision_M'], 'r-.|', markevery=[best_model_epoch])    
    plt.ylabel('Measure')
    plt.xlabel('Epoch')
    plt.legend(['Train Recall Micro', 'Train Precision Micro', 'Train Recall Macro', 'Train Precision Macro', \
                'Val Recall Micro', 'Val Precision Micro', 'Val Recall Macro', 'Val Precision Macro'], loc='lower left')
    plt.xlim([0, last_epoch])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_prec_rec.png')
    plt.close()

    #plot precision and recall by class on the validation set
    plt.plot(history.history['val_recall_0'], 'y-|', markevery=[best_model_epoch])
    plt.plot(history.history['val_precision_0'], 'y--|', markevery=[best_model_epoch])
    plt.plot(history.history['val_recall_1'], 'm-|', markevery=[best_model_epoch])
    plt.plot(history.history['val_precision_1'], 'm--|', markevery=[best_model_epoch])
    plt.plot(history.history['val_recall_2'], 'g-|', markevery=[best_model_epoch])
    plt.plot(history.history['val_precision_2'], 'g--|', markevery=[best_model_epoch])
    plt.title('Model Precision and Recall')
    plt.ylabel('Measure')
    plt.xlabel('Epoch')
    plt.legend(['Reca B', 'Prec B', 'Reca N', 'Prec N', 'Reca M', 'Prec M'], loc='lower left')#B=bckgrnd, N=neur, M=mesen
    plt.xlim([0, last_epoch])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_pr_classes.png')
    plt.close()

    #plot micro and macro f1 evaluated on the training and validation set every epoch (not minibatch)
    plt.plot(history.history['f1_micro'], 'C0-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_micro'], 'C1-o', markevery=[best_model_epoch])
    plt.plot(history.history['f1_macro'], 'C0--o', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_macro'], 'C1--o', markevery=[best_model_epoch])
    plt.title('F1 Scores on Epochs')
    plt.ylabel('Measure')
    plt.xlabel('Epoch')
    plt.legend(['Train F1 Micro', 'Val F1 Micro', 'Train F1 Macro', 'Val F1 Macro'], loc='lower left')
    plt.xlim([0, last_epoch])
    plt.ylim(0,1)
    plt.savefig(model_name + 'f1_epoch.png')
    plt.close()

    


def save_summary_txt(model, history, filename, history_index=-1):
    val_f1 = history.history['val_f1_m']
    val_recall = history.history['val_recall_m']
    val_precision = history.history['val_precision_m']
    val_acc = history.history['val_accuracy']
    val_bin_acc = history.history['val_binary_accuracy']

    f1 = history.history['f1_m']
    recall = history.history['recall_m']
    precision = history.history['precision_m']
    acc = history.history['accuracy']
    bin_acc = history.history['binary_accuracy']
    
    filename_txt = filename + '.txt'
    f = open(filename_txt, 'a')
    output_text = 'Validation Set Metrics:\n\n' \
        + 'Val F1:\t' + str(val_f1[history_index]) \
        + '\n' +  'Val Recall:\t' + str(val_recall[history_index]) \
        + '\n' +  'Val Precision:\t' + str(val_precision[history_index]) \
        + '\n' +  'Val Accuracy:\t' + str(val_acc[history_index]) \
        + '\n' +  'Val Binary Accuracy:\t' + str(val_bin_acc[history_index]) \
        \
        + '\n\nTraining Set Metrics:\n\n' \
        + 'F1:\t\t' + str(f1[history_index]) \
        + '\n' +  'Recall:\t\t' + str(recall[history_index]) \
        + '\n' +  'Precision:\t' + str(precision[history_index]) \
        + '\n' +  'Accuracy:\t' + str(acc[history_index]) \
        + '\n' +  'Binary Accuracy:\t' + str(bin_acc[history_index])
        
    f.write(output_text)

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = '\n'.join(stringlist)
    f.writelines('\n\nModel summary:  ' + filename[filename.rfind('/')+1:]  +'\n\n')
    f.writelines(short_model_summary + '\n\n')
    f.close()

# save the learning curves, summary, and model weights in a new folder
def save_model(model, history, last_epoch, best_model_epoch, model_name, path):
    subdir = path + model_name + '/'
    try:
        os.mkdir(subdir)
    except FileExistsError:
        pass
    full_filename = subdir + model_name
    save_summary_txt(model, history, full_filename, last_epoch)
    plot_learning_curves(history, last_epoch, best_model_epoch, full_filename)
    model.save(full_filename + '.h5')
