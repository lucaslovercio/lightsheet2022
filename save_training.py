import matplotlib.pyplot as plt
# the commented lines below are a possible solution to thread issues caused by tkinter
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
import os
import numpy as np
import skimage.io as io
#TODO
'''
- need to add some documentation
- consider changing what color and linetypes are used to make the learning curves
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
    plt.plot(history.history['binary_accuracy_batch'], 'C1-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_binary_accuracy_batch'], 'C1--o', markevery=[best_model_epoch])
    plt.plot(history.history['tissue_type_accuracy_batch'], 'C2-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_tissue_type_accuracy_batch'], 'C2--o', markevery=[best_model_epoch])
    plt.title('Model Accuracy Batch Averaged')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Acc', 'Val Acc', 'Train Binary Acc', 'Val Binary Acc', 'Train Tissue Acc', 'Val Tissue Acc'], loc='lower right')
    plt.xlim([0,last_epoch])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_acc.png')
    plt.close()

    # plot training and validation accuracy values
    plt.plot(history.history['accuracy'], 'C0-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_accuracy'], 'C0--o', markevery=[best_model_epoch])
    plt.plot(history.history['binary_accuracy'], 'C1-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_binary_accuracy'], 'C1--o', markevery=[best_model_epoch])
    plt.plot(history.history['tissue_type_accuracy'], 'C2-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_tissue_type_accuracy'], 'C2--o', markevery=[best_model_epoch])
    plt.title('Model Accuracy Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Acc', 'Val Acc', 'Train Binary Acc', 'Val Binary Acc', 'Train Tissue Acc', 'Val Tissue Acc'], loc='lower right')
    plt.xlim([0,last_epoch])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_epochacc.png')
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

    # plot batch averaged precision, recall, and f1 score
    plt.plot(history.history['f1_macro_batch'], 'C0--o', markevery=[best_model_epoch])
    plt.plot(history.history['recall_macro_batch'], 'C1--o', markevery=[best_model_epoch])
    plt.plot(history.history['precision_macro_batch'], 'C2--o', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_macro_batch'], 'C0-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_recall_macro_batch'], 'C1-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_precision_macro_batch'], 'C2-o', markevery=[best_model_epoch])
    plt.title('Batch Val/Train F1')
    plt.ylabel('Measure')
    plt.xlabel('Epoch')
    plt.legend(['Train F1', 'Train Reca', 'Train Prec', 'Val F1', 'Val Reca', 'Val Prec'])
    plt.xlim([0, last_epoch])
    plt.ylim(0,1)
    plt.savefig(model_name + '_f1.png')
    plt.close()

    
    # # plot training and validation f1 scores
    # plt.plot(history.history['f1_m'], 'b-|', markevery=[best_model_epoch])
    # plt.plot(history.history['f1_M'], 'b--|', markevery=[best_model_epoch])
    # plt.plot(history.history['val_f1_m'], 'r-|', markevery=[best_model_epoch])
    # plt.plot(history.history['val_f1_M'], 'r--|', markevery=[best_model_epoch])
    # plt.title('Model F1 Metrics')
    # plt.ylabel('F1')
    # plt.xlabel('Epoch')
    # plt.legend(['Train F1 Micro', 'Train F1 Macro', 'Val F1 Micro', 'Val F1 Macro'], loc='lower left')
    # plt.xlim([0, last_epoch])
    # plt.ylim(0,1)
    # #plt.show()
    # plt.savefig(model_name + '_f1.png')
    # plt.close()

    # # plot in depth recall and precision for validation and training TODO should cut this graph
    # plt.plot(history.history['recall_m'], 'b-|', markevery=[best_model_epoch])
    # plt.plot(history.history['precision_m'], 'b--|', markevery=[best_model_epoch])
    # plt.plot(history.history['recall_M'], 'b:|', markevery=[best_model_epoch])
    # plt.plot(history.history['precision_M'], 'b-.|', markevery=[best_model_epoch])    
    # plt.plot(history.history['val_recall_m'], 'r-|', markevery=[best_model_epoch])
    # plt.plot(history.history['val_precision_m'], 'r--|', markevery=[best_model_epoch])
    # plt.plot(history.history['val_recall_M'], 'r:|', markevery=[best_model_epoch])
    # plt.plot(history.history['val_precision_M'], 'r-.|', markevery=[best_model_epoch])    
    # plt.ylabel('Measure')
    # plt.xlabel('Epoch')
    # plt.legend(['Train Recall Micro', 'Train Precision Micro', 'Train Recall Macro', 'Train Precision Macro', \
    #             'Val Recall Micro', 'Val Precision Micro', 'Val Recall Macro', 'Val Precision Macro'], loc='lower left')
    # plt.xlim([0, last_epoch])
    # plt.ylim(0,1)
    # #plt.show()
    # plt.savefig(model_name + '_prec_rec.png')
    # plt.close()

    #plot batch precision and recall by class on the validation set
    plt.plot(history.history['val_reca_0'], 'C0-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_prec_0'], 'C0--o', markevery=[best_model_epoch])
    plt.plot(history.history['val_reca_1'], 'C1-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_prec_1'], 'C1--o', markevery=[best_model_epoch])
    plt.plot(history.history['val_reca_2'], 'C2-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_prec_2'], 'C2--o', markevery=[best_model_epoch])
    plt.title('Val Batch Classwise Prec/Recc')
    plt.ylabel('Measure')
    plt.xlabel('Epoch')
    plt.legend(['Reca B', 'Prec B', 'Reca N', 'Prec N', 'Reca M', 'Prec M'], loc='lower left')#B=bckgrnd, N=neur, M=mesen
    plt.xlim([0, last_epoch])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_batchpr_classes.png')
    plt.close()

    #plot epoch precision and recall by class on the validation set
    plt.plot(history.history['val_recall_macro_0'], 'C0-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_precision_macro_0'], 'C0--o', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_macro_0'], 'C0:o', markevery=[best_model_epoch])
    plt.plot(history.history['val_recall_macro_1'], 'C1-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_precision_macro_1'], 'C1--o', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_macro_1'], 'C1:o', markevery=[best_model_epoch])
    plt.plot(history.history['val_recall_macro_2'], 'C2-o', markevery=[best_model_epoch])
    plt.plot(history.history['val_precision_macro_2'], 'C2--o', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_macro_2'], 'C2:o', markevery=[best_model_epoch])
    plt.title('Val Epochwise Classwise Precision/Recall')
    plt.ylabel('Measure')
    plt.xlabel('Epoch')
    plt.legend(['Reca B', 'Prec B', 'F1 B', 'Reca N', 'Prec N', 'F1 N', 'Reca M', 'Prec M', 'F1 M'], loc='lower left')#B=bckgrnd, N=neur, M=mesen
    plt.xlim([0, last_epoch])
    plt.ylim(0,1)
    #plt.show()
    plt.savefig(model_name + '_pr_classes.png')
    plt.close()
    
    #plot macro f1 evaluated on the training and validation set every epoch (not minibatch)
    plt.plot(history.history['f1_macro'], 'C0-o', markevery=[best_model_epoch])
    # plt.plot(history.history['recall_macro'], 'C1--o', markevery=[best_model_epoch])
    # plt.plot(history.history['precision_macro'], 'C2--o', markevery=[best_model_epoch])
    plt.plot(history.history['val_f1_macro'], 'C1-o', markevery=[best_model_epoch])
    # plt.plot(history.history['val_recall_macro'], 'C1-o', markevery=[best_model_epoch])
    # plt.plot(history.history['val_precision_macro'], 'C2-o', markevery=[best_model_epoch])
    
    plt.title('F1 Scores on Epochs')
    plt.ylabel('Measure')
    plt.xlabel('Epoch')
    plt.legend(['Train F1', 'Val F1'], loc='upper left')
    plt.xlim([0, last_epoch])
    plt.ylim(0.5,1)
    plt.savefig(model_name + '_f1_epoch.png')
    plt.close()

    


def save_summary_txt(model, history, filename, model_info, history_index=-1, val_history={}, test_history={}):#TODO1
    filename_txt = filename + '.txt'
    f = open(filename_txt, 'a')
    output_text = '\tMetrics From Training History:\n\n'
    for key in history.history.keys():
        output_text += key + ':\t\t' + str(history.history[key][history_index]) + '\n'
    if bool(val_history):
        output_text += '\n\tPost-Training Validation Metrics:\n\n'
        for key in val_history.keys():
            output_text += key + ':\t\t' + str(val_history[key]) + '\n'
    if bool(test_history):
        output_text += '\n\tMetrics from the Test Set:\n\n'
        for key in test_history.keys():
            output_text += key + ':\t\t' + str(test_history[key]) + '\n'
    f.write(output_text)
    
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = '\n'.join(stringlist)
    f.writelines('\n\nModel summary:  ' + model_info + '\n\n')#filename[filename.rfind('/')+1:]  +'\n\n')#TODO1
    f.writelines(short_model_summary + '\n\n')
    f.close()

# save the learning curves, summary, and model weights in a new folder
def save_model(model, history,last_epoch, best_model_epoch, model_name, model_info, path, val_history={}, test_history={}):#TODO1
    subdir = path + model_name + '/'
    try:
        os.mkdir(subdir)
    except FileExistsError:
        pass
    full_filename = subdir + model_name
    save_summary_txt(model, history, full_filename, model_info, last_epoch, val_history, test_history)#TODO1
    plot_learning_curves(history, last_epoch, best_model_epoch, full_filename)
    model.save(full_filename + '.h5')
