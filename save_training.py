import matplotlib.pyplot as plt
import os
import numpy as np
import skimage.io as io
import cv2
#TODO
'''
- change showLosses so that it plots the new metrics
- understand and improve the text files being saved
- clean this up
'''


Foreground = [250,250,50]
Background = [0,150,0]
COLOR_DICT = np.array([Foreground, Background])

def labelVisualize(num_class,color_dict,img):
    #vectorUnique = np.unique(img)
    #print(vectorUnique)
    #print("shape " + str(img.shape) + " type " + str(type(img)))
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    #print("out img shape " + str(img.shape) + " type " + str(type(img)))
    img_out = img_out.astype("uint8")
    #for i in range(num_class):
    #    img_out[img == i,:] = color_dict[i]
    img_out[img > 0.3, :] = color_dict[0]
    img_out[img <= 0.3, :] = color_dict[1]
    return img_out

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        print("val img " + str(i))
        img = labelVisualize(num_class,COLOR_DICT,item) #if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

def saveHistory(model, history, metrics, filename):
    ''' guarda en un .txt los valores que se obtuvieron en el entrenamiento '''
    ''' save in a .txt the values obtained during training  '''

    name = filename + '_history.txt'
    arch = open(name, 'w')

    # obtengo los valores guardados en el history generado del entrenamiento
    # get the values saved in the history generated from training
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fscoreM_val = metrics.val_fscoreM

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    arch.writelines('Model summary:  ' + filename  +'\n' + '\n')

    arch.writelines(short_model_summary + '\n' + '\n')

    arch.writelines('\n' +'---------------------------------------------'+ '\n')

    arch.writelines('ACC            :  ' + repr(acc) + '\n')
    arch.writelines('VALIDATION_ACC:' + repr(val_acc) + '\n' + '\n')
    arch.writelines('LOSS           :' + repr(loss) + '\n')
    arch.writelines('VALIDATION_LOSS:' + repr(val_loss) + '\n' + '\n')
    arch.writelines('F1 M VAL:' + repr(fscoreM_val) + '\n' + '\n')


    arch.close()
    return val_loss[-1]

#TODO this function is where the new metrics will be added to plot them
#def showLosses(history,metrics,epochs,modelName):
def showLosses(history, epochs, modelName):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    #plt.plot(history.history['jaccard_coef'])
    #plt.plot(history.history['val_jaccard_coef'])
    #plt.plot(history.history['dice_coef'])
    #plt.plot(history.history['val_dice_coef'])
    # plt.plot(metrics.val_fscoreM)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    #plt.legend(['Acc Train', 'Acc Val', 'FscoreM Val'], loc='upper left')
    plt.legend(['Acc Train', 'Acc Val', 'JM Train', 'JM Val', 'Dice Train', 'Dice Val'], loc='upper left')
    plt.xlim([0,epochs])
    plt.ylim(0,1)
    #plt.show()

    plt.savefig(modelName + '_acc.png')
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower left')
    plt.xlim([0,epochs])
    plt.ylim(0,3)
    #plt.show()
    plt.savefig(modelName + '_losses.png')
    plt.close()

    # Plot training & validation loss values
    #metrics.recall_m,metrics.precision_m,metrics.f1_m
    #plt.plot(history.history['val_recall_m'])
    #plt.plot(history.history['val_precision_m'])
    #plt.plot(history.history['val_f1_m'])
    #plt.title('Model F1 metrics')
    #plt.ylabel('F1')
    #plt.xlabel('Epoch')
    #plt.legend(['Val_Reca','Val_prec','Val_F1'], loc='lower left')
    #plt.xlim([0,epochs])
    #plt.ylim(0,1)
    #plt.show()
    #plt.savefig(modelName + '_f1.png')
    #plt.close()

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

#TODO find out if this actually gets used anywhere
def prediction(model,imagePath, IMG_HEIGHT, IMG_WIDTH,image_output):
    img = cv2.imread(imagePath, 0)
    #train_img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE) / 255.
    img = cv2.resize(img, (int(IMG_WIDTH), int(IMG_HEIGHT)))
    img = np.expand_dims(img, axis=-1)
    x_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    # testimg=resize(img,(self.IMG_HEIGHT,self.IMG_WIDTH),mode='constant',preserve_range=True)
    x_test[0] = img
    preds_test = model.predict(x_test, verbose=1)

    preds_test = (preds_test > 0.5).astype(np.uint8)
    mask = preds_test[0]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 1:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    merged_image = cv2.merge((mask, mask, mask))

    cv2.imwrite(image_output, merged_image)

    return x_test[0], mask

def saveModelSummary(model, filename):
    ''' guarda en un .txt los valores que se obtuvieron en el entrenamiento '''
    name = filename + '_model.txt'
    arch = open(name, 'w')
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    arch.writelines('Model summary:  ' + filename  +'\n' + '\n')
    arch.writelines(short_model_summary + '\n' + '\n')
    arch.close()
