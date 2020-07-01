import matplotlib.pyplot as plt
import os
import numpy as np
import skimage.io as io
import cv2
#TODO
'''
- will need to add the new metrics to show_losses once they are ready
- understand and improve the text files being saved
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

    # plot training & validation accuracy values
    plt.plot(history.history['accuracy'], marker='o', markevery=[best_model_epoch])
    plt.plot(history.history['val_accuracy'], marker='o', markevery=[best_model_epoch])    
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Acc Train', 'Acc Val'], loc='upper left')
    plt.xlim([0,total_epochs])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_acc.png')
    plt.close()

    # plot training & validation loss values
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

    # plot training & validation loss values
    plt.plot(history.history['val_recall_m'], marker='o', markevery=[best_model_epoch])
    plt.plot(history.history['val_precision_m'], marker='o', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_m'], marker='o', markevery=[best_model_epoch])
    plt.title('Model F1 metrics')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Val Rec','Val Prec','Val F1'], loc='lower left')
    plt.xlim([0, total_epochs])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_f1.png')
    plt.close()

def saveLossesTXT(history,modelNameFull):
    vF1 = history.history['val_f1_m']
    vReca = history.history['val_recall_m']
    vPrec = history.history['val_precision_m']
    vAcc = history.history['val_accuracy']
    filenameTxt = modelNameFull + ".txt"
    f = open(filenameTxt, "a")
    strToSave = "F1 val:" + str(vF1[-1]) + "\n" +  "Reca val:" + str(vReca[-1]) + "\n" +  "Prec val:" + str(vPrec[-1]) \
                + "\n" +  "Acc val:" + str(vAcc[-1])
    f.write(strToSave)
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
