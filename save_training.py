import matplotlib.pyplot as plt
import os
import numpy as np
import skimage.io as io
import cv2
#TODO
'''
- fix up save models txt
- collapse the model summary and saved text file to the same folder
- maybe change the saving method so that it saves all the output for a given model in its own folder
- need to add some documentation
'''

'''
Name: show_losses
Purpose:
This function creates and saves plots of model performance metrics.
Arguments:
- history:
 The training history Keras saves during the call to model.fit.
- total_epochs:
 The total number of epochs the model was run for.
- best_model_epoch:
 The epoch at which the model's weights was done training. Early stopping caused several 
 epochs to run after this point, which is marked with a dot on the plots.
- model_name:
 The full filepath to and name of the model.
Outputs:
Saves plots to the directory with the model weights.
'''
def show_losses(history, total_epochs, best_model_epoch, model_name):

    # plot training and validation accuracy values
    plt.plot(history.history['accuracy'], marker='o', markevery=[best_model_epoch])
    plt.plot(history.history['val_accuracy'], marker='o', markevery=[best_model_epoch])
    plt.plot(history.history['background_accuracy'], marker='o', markevery=[best_model_epoch])# new
    plt.plot(history.history['val_background_accuracy'], marker='o', markevery=[best_model_epoch])# new
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Acc', 'Val Acc', 'Train Binary Acc', 'Val Binary Acc'], loc='upper left')# changed
    plt.xlim([0,total_epochs])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_acc.png')
    plt.close()

    # plot training and validation loss values
    plt.plot(history.history['loss'], marker='o', markevery=[best_model_epoch])
    plt.plot(history.history['val_loss'], marker='o', markevery=[best_model_epoch])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Val Loss'], loc='lower left')
    plt.xlim([0,total_epochs])
    plt.ylim(0, max(history.history['loss']) + 1)
    #plt.show()
    plt.savefig(model_name + '_losses.png')
    plt.close()

    # plot training and validation f1 scores
    plt.plot(history.history['f1_m'], marker='o', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_m'], marker='o', markevery=[best_model_epoch])
    plt.title('Model F1 metrics')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Train F1', 'Val F1'], loc='lower left')
    plt.xlim([0, total_epochs])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_f1.png')
    plt.close()
    

#TODO1 add in new metrics
def save_losses_txt(history, filename, history_index):
    val_f1 = history.history['val_f1_m']
    val_recall = history.history['val_recall_m']
    val_precision = history.history['val_precision_m']
    val_acc = history.history['val_accuracy']

    f1 = history.history['f1_m']
    recall = history.history['recall_m']
    precision = history.history['precision_m']
    acc = history.history['accuracy']
    
    filename_txt = filename + '.txt'
    f = open(filename_txt, 'a')
    output_text = 'Validation Set Metrics:\n\n' \
        + 'Val F1:\t' + str(val_f1[-1]) \
        + '\n' +  'Val Recall:\t' + str(val_recall[history_index]) \
        + '\n' +  'Val Precision:\t' + str(val_precision[history_index]) \
        + '\n' +  'Val Accuracy:\t' + str(val_acc[history_index]) \
        \
        + '\n\nTraining Set Metrics:\n\n' \
        + 'F1:\t\t' + str(f1[-1]) \
        + '\n' +  'Recall:\t\t' + str(recall[history_index]) \
        + '\n' +  'Precision:\t' + str(precision[history_index]) \
        + '\n' +  'Accuracy:\t' + str(acc[history_index]) \

        
    f.write(output_text)
    f.close()

def saveModelSummary(model, filename):
    ''' save in a .txt file the values obtained from training'''
    name = filename + '_model.txt'
    arch = open(name, 'w')
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    arch.writelines('Model summary:  ' + filename  +'\n' + '\n')
    arch.writelines(short_model_summary + '\n' + '\n')
    arch.close()
