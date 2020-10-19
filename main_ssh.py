# imports
import os
import pickle
import time
import imp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import json

import seaborn as sns

from gensim.models import Word2Vec

from keras import backend as K
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from keras.utils import to_categorical   


from MoreFunctions.code import preprocess
from MoreFunctions.code import models

from MoreFunctions.innvestigate import create_analyzer

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score

from keras.callbacks import EarlyStopping

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from zipfile import ZipFile 

import tensorflow as tf
import tensorflow_hub as hub
import re
# import bert
# from bert.tokenization import FullTokenizer
from tqdm import tqdm_notebook
from tensorflow.keras import backend as K

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold

import nltk
import spacy
# !python -m spacy download en_core_web_sm
nlp_spacy = spacy.load('en_core_web_sm')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


from data_preprocessed import generate_cnndata
from training_model import textCNNwithoutembedding, build_modelMLP
from xai_functions import generateanalisys, plot_text_heatmap

sess = tf.Session()

# dataset
# drive_dir = 'database/'
path = 'MoreFunctions/' 
datapath1 = 'database/software_posts_cognitive.xlsx' #software engineering course (1747 messages)
datapath2 =  'database/lct_agree_cognitive.xlsx' #logical and critical thinking MOOC set (1479 messages)

# # folders for weights and results of training and testing on both software course and LCT MOOC post set
# drive_dir_weights =  'weights_both/'
# drive_dir_results =  'results_both/'

# # folders for weights and results of training and testing on software course
# drive_dir_weights =  'weights_soft/'
# drive_dir_results =  'results_soft/'

# folders for weights and results of training and testing on LCT MOOC
drive_dir_weights =  'weights_results/weights/'
drive_dir_results =  'weights_results/results/'

# use pandas to load the data from the software course
# originaldata = pd.read_excel(datapath1)

# # use pandas to load the data from the LCT MOOC
# originaldata = pd.read_excel(datapath2)

#load both datasets
data1 = pd.read_excel(datapath1)[['phaseId','postBody']]
data1['courseType'] = 1             #software engineering
data2 = pd.read_excel(datapath2)[['phaseId','postBody']]
data2['courseType'] = 2             #LCT MOOC

# #combine two dfs / shuffle the rows / reindex 3226 messages in total
originaldata = pd.concat([data1,data2])
# originaldata = originaldata.sample(frac=1, random_state=1)
originaldata.reset_index(drop=True, inplace=True)

# getting the class
classes = originaldata['phaseId'].tolist()

# changing the format of the class to categorical
encodedclass = to_categorical(classes)
# IDs = originaldata['courseID'].tolist() # courseID in the software dataset
# IDs = originaldata['postSectionId'].tolist() # sectionID in the LCT MOOC dataset
# getting the texts
texts = originaldata['postBody'].tolist()

# getting the courseType
courseType= originaldata['courseType'].tolist()

# print(len(classes))
print(len(encodedclass))
originaldata.info()
'''
########################## pre processing #########################
# Loading the glove into a word2vec gensim object
word2vec_filenametxt = 'glovefiles/' + 'glove.6B.100d.txt'
word2vec_filename = 'glovefiles/' + 'glove.6B.100d.txt.word2vec'
# word2vec_filename = 'glovefiles/' + 'glove.6B.100d.w2vformat.txt'
modelglove = KeyedVectors.load_word2vec_format(word2vec_filename, binary=False)
# modelglove = KeyedVectors.load_word2vec_format(word2vec_filenametxt, binary=False)
print('glove loaded.')
print('text converting...')

newdata, data, preprocessed, heigth ,width, glovemodel = generate_cnndata(texts, model = modelglove,  path = word2vec_filename)
print('preprocessed finished.')

# software course index from 0 to 1746
course1_set_new = newdata[0:1747,:,:]
course1_class = classes[0:1747]
# lct course index from 1747 to 3225
course2_set_new = newdata[1747:3226,:,:]
course2_class = classes[1747:3226]

print("course 1 new set:")
print(course1_set_new.shape)
print(len(course1_class))

print("course 2 new set:")
print(course2_set_new.shape)
print(len(course2_class))

########################## Split data before the training loop #########################

############## EXPERIMENT 5 ########
# ###### no.1  validation set of 0.1 from software course (course 1) 
# indices = np.arange(len(course1_set_new))
# x_train_1, x_valid, y_train_1, y_valid, index_train_1,index_valid = train_test_split(course1_set_new, course1_class, indices, stratify=course1_class, random_state = 1 ,test_size=0.1)

# print("after splitting...")
# print(x_train_1.shape)
# print(len(y_train_1))
# print(x_valid.shape)
# print(len(y_valid))

# print("combining rest of the data...")
# # combine the rest set as training set together into the cv loop 
# cv_classes = []
# cv_set = np.concatenate((x_train_1, course2_set_new), axis = 0)
# cv_classes = y_train_1

# for item in course2_class:
#   cv_classes.append(item)

# ###### no.2 validation set of 0.1 from lct course (course 2) 
# indices = np.arange(len(course2_set_new))
# x_train_2, x_valid, y_train_2, y_valid, index_train_2,index_valid = train_test_split(course2_set_new, course2_class, indices, stratify=course2_class, random_state = 1 ,test_size=0.1)

# print("after splitting...")
# print(x_train_2.shape)
# print(len(y_train_2))
# print(x_valid.shape)
# print(len(y_valid))

# print("combining rest of the data...")
# # combine the rest set as training set together into the cv loop 
# cv_classes = []
# cv_set = np.concatenate((course1_set_new,x_train_2,), axis = 0)
# cv_classes = course1_class

# for item in y_train_2:
#   cv_classes.append(item)


# print("after combining...")
# print(cv_set.shape)
# print(len(cv_classes))

# # get encodedcategorical data 
# cv_encodedclass = to_categorical(cv_classes)

# # save the x y_valid for the testing work after the cv selection
# np.save("validation/x_validation_ex5lct.npy",x_valid)
# np.save("validation/y_validation_ex5lct.npy",y_valid)
# np.save("validation/index_validation_ex5lct.npy",index_valid) # index_valid is the index in the individual course2 (lct ) set

# print("validation data " + str(len(x_valid)) + " saved.")
# # save the training data for cv loop
# np.save("validation/cv_set_ex5lct.npy",cv_set)
# np.save("validation/cv_classes_ex5lct.npy",cv_classes)
# print("training (CV) data " + str(len(cv_set)) + " saved.")

# ###### no.3 validation set of 0.1 from both sets
indices_1 = np.arange(len(course1_set_new))
indices_2 = np.arange(len(course2_set_new))
x_train_1, x_valid_1, y_train_1, y_valid_1, index_train_1,index_valid_1 = train_test_split(course1_set_new, course1_class, indices_1, stratify=course1_class, random_state = 1 ,test_size=0.1)
x_train_2, x_valid_2, y_train_2, y_valid_2, index_train_2,index_valid_2 = train_test_split(course2_set_new, course2_class, indices_2, stratify=course2_class, random_state = 1 ,test_size=0.1)

print("after splitting...")
print("course 1 training set x:")
print(x_train_1.shape)
print("course 2 training set x:")
print(x_train_2.shape)
print("course 1 training set y:")
print(len(y_train_1))
print("course 2 training set y:")
print(len(y_train_2))
print("course 1 validation set x:")
print(x_valid_1.shape)
print("course 1 validation set y:")
print(len(y_valid_1))
print("course 2 validation set x:")
print(x_valid_2.shape)
print("course 2 validation set y:")
print(len(y_valid_2))

print("combining the training and validation set of the data...")
# combine the  training set together into the cv loop 
cv_classes = []
cv_set = np.concatenate((x_train_1,x_train_2), axis = 0)

cv_classes = y_train_1

for item in y_train_2:
  cv_classes.append(item)


print("after combining the training set...")
print(cv_set.shape)
print(len(cv_classes))

# get encodedcategorical data 
cv_encodedclass = to_categorical(cv_classes)


# combine the  validation set together
y_valid = []
x_valid = np.concatenate((x_valid_1,x_valid_2), axis = 0)

y_valid = y_valid_1

for item in y_valid_2:
  y_valid.append(item)

print("after combining the validation set...")
print(x_valid.shape)
print(len(y_valid))

# save the x y_valid for the testing work after the cv selection
np.save("validation/x_validation1_ex5both.npy",x_valid_1)
np.save("validation/y_validation1_ex5both.npy",y_valid_1)
np.save("validation/index_validation1_ex5both.npy",index_valid_1) # index_valid is the index in the individual course1 (software ) set

np.save("validation/x_validation2_ex5both.npy",x_valid_2)
np.save("validation/y_validation2_ex5both.npy",y_valid_2)
np.save("validation/index_validation2_ex5both.npy",index_valid_2) # index_valid is the index in the individual course2 (lct) set

print("validation data 1 " + str(len(x_valid_1)) + " and validation data 2 " + str(len(x_valid_2)) + " saved.")

np.save("validation/x_validation_ex5both.npy",x_valid)
np.save("validation/y_validation_ex5both.npy",y_valid)

print("validation data in total " + str(len(x_valid)) + " and " + str(len(y_valid)) + " saved.")

# save the training data for cv loop
np.save("validation/cv_set_ex5both.npy",cv_set)
np.save("validation/cv_classes_ex5both.npy",cv_classes)
print("training (CV) data " + str(len(cv_set)) + " saved.")


########################## training (n-fold cv) #########################

verb = False
patience = 100
sizekfold = 10
batch_size = 36
epochs = 300
filter_sizes = [1,2,3,5,7] #default[1,2,3,5]
num_filters = 36 #default36
activ = 'softmax'
nclasses = 5
losstype = 'categorical_crossentropy'


def calculatemetrics(y_orig,y_pred):
  
  resultados = {}
  
  resultados['accuracy'] = accuracy_score(y_orig, y_pred)
  resultados['precision'] = precision_score(y_orig, y_pred, average=None)
  resultados['recall'] = recall_score(y_orig, y_pred,average=None)
  resultados['f1'] = f1_score(y_orig, y_pred,average='weighted')
  resultados['kappa'] = cohen_kappa_score(y_orig, y_pred)

  return resultados

#fileheatmap = drive_dir_heatmaps + adtext + '_heatmapopinion.txt'
print("start training...")

for seedkfold in [1,2,3,4,5,6]:
  
  kfold = StratifiedKFold(sizekfold,True,seedkfold)
  foldnumber = 1

  # for train_index,test_index in kfold.split(newdata,classes):
  for train_index,test_index in kfold.split(cv_set,cv_classes):
    K.clear_session()

    adtext = 'Round' + str(seedkfold) + 'Fold' + str(foldnumber)
    filename = drive_dir_results + adtext +'_result.csv'
    # filename = drive_dir_resultados + adtext +'_result.csv'
    
    if os.path.exists(filename):
      print("Already executed: "+ filename)
      foldnumber += 1
      continue

    filepath = drive_dir_weights + adtext + '_model.weights.best.hdf5'
    # filepath = drive_dir_pesos + adtext + '_model.weights.best.hdf5' #LCT MOOC training path
    checkpointer = models.getmodelcheckpoint(filepath,verbose = 0)
  
    # x_train, x_test = np.array(newdata)[train_index],np.array(newdata)[test_index]
    # y_train, y_test, y_tousemetrics = np.array(encodedclass)[train_index], np.array(encodedclass)[test_index], np.array(classes)[test_index]
    x_train, x_test = np.array(cv_set)[train_index],np.array(cv_set)[test_index]
    y_train, y_test, y_tousemetrics = np.array(cv_encodedclass)[train_index], np.array(cv_encodedclass)[test_index], np.array(cv_classes)[test_index]

    # model,modelnoactiv = textCNNwithoutembedding(height=newdata.shape[1],width=newdata.shape[2],classes=nclasses,activ=activ,filter_sizes=filter_sizes,num_filters=num_filters)
    model,modelnoactiv = textCNNwithoutembedding(height=cv_set.shape[1],width=cv_set.shape[2],classes=nclasses,activ=activ,filter_sizes=filter_sizes,num_filters=num_filters)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=patience)

    start = time.time()
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),callbacks=[checkpointer, es], verbose=verb)
    end = time.time()

    epocc = len(hist.history['loss'])

    model.load_weights(filepath)
    modelnoactiv.load_weights(filepath)
    
    predictions = model.predict(x_test)
    y_pred = [np.argmax(lista) for lista in predictions]

    resultados = calculatemetrics(y_tousemetrics,y_pred)

    print("---------")
    print(adtext + ":")
    print(resultados)

    # save in the csvs
    fo = open(filename, 'w')
    fo.write(
        'fold,activation,accuracy_te,precision_te,recall_te,f1_te,kappa_te,train_time,train_epocs\n')
    fo.close()

    fo = open(filename, 'a')
    fo.write( 
        str(foldnumber) +  ',' + str(activ)  +   ',' + str(resultados['accuracy']) +
        ',' + str(resultados['precision']) + ',' + str(resultados['recall']) + ',' + str(resultados['f1']) + ',' + str(resultados['kappa']) + ',' + 
        str(end-start) + ',' + str(epocc)  + ',' + '\n')
    fo.close()
    
    foldnumber += 1



########################## training result matrices #########################
dictresults = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'kappa': []}

# Geting the values of the metrics in every fold 
for seedkfold in [1,2,3,4,5,6]:
  for foldnumber in range(1,11):
    adtext = 'Round' + str(seedkfold) + 'Fold' + str(foldnumber)
    filename = drive_dir_results + adtext +'_result.csv'
    # Maybe will throw an error here, because i dont know if i'm reading 
    # properly the csv (need to set the index_col as False in this case)
    readedfile = pd.read_csv(filename, index_col=False, header = 0)

    dictresults['acc'].append(readedfile['accuracy_te'])
    dictresults['precision'].append(list(readedfile['precision_te']))
    dictresults['recall'].append(list(readedfile['recall_te']))
    dictresults['f1'].append(readedfile['f1_te'])
    dictresults['kappa'].append(readedfile['kappa_te'])

#average precision 
precision_avg = [0, 0, 0, 0, 0]
for item in dictresults['precision']:
  item = item[0]
  item = item.strip('[')
  item = item.strip(']')
  item = item.split(' ')
  item = [float(x) for x in item if x != '']
  for j in range(5):
    precision_avg[j] += item[j]

precision_avg = [i/len(dictresults['precision']) for i in precision_avg]

#average recall 
recall_avg = [0, 0, 0, 0, 0]
for item in dictresults['recall']:
  item = item[0]
  item = item.strip('[')
  item = item.strip(']')
  item = item.split(' ')
  item = [float(x) for x in item if x != '']
  for j in range(5):
    recall_avg[j] += item[j]

recall_avg = [i/len(dictresults['recall']) for i in recall_avg]

# Printing the mean for the 5 metrics
print('average accuracy: ' + str(np.mean(dictresults['acc'])))
print('average kappa score: ' + str(np.mean(dictresults['kappa'])))
print('average f1 score: ' + str(np.mean(dictresults['f1'])))
print('average precision: ' + str(precision_avg))
print('average recall: ' + str(recall_avg))
print('---------')
print('maximum accuracy: ' + str(np.max(dictresults['acc'])))
print('maximum kappa: ' + str(np.max(dictresults['kappa'])))
print('maximum f1 score: ' + str(np.max(dictresults['f1'])))

df_result = pd.DataFrame.from_dict(dictresults, orient='index')

#results of testing on soft datasets
# df_result.to_csv( "weights_results/result_ex5_tsoft1.csv", index = True, header=True)

# #results of testing on soft datasets
# df_result.to_csv( "weights_results/result_ex5_tlct1.csv", index = True, header=True)

#results of testing on both datasets but split before cv loop
df_result.to_csv( "weights_results/result_ex5_both.csv", index = True, header=True)

# #results of lct datasets
# df_result.to_csv(drive_dir+ "result_lct.csv", index = True, header=True)

# # results of training on both sets
# df_result.to_csv("result_both.csv", index = True, header=True)
print(df_result)

'''
########################## test on the model with the best weights #########################
# This part is for the hold out cross validation experiment
# so you dont use this one tu get the mean results of your kfold cross validation part
def calculatemetrics(y_orig,y_pred):
  
  resultados = {}
  
  resultados['accuracy'] = accuracy_score(y_orig, y_pred)
  resultados['precision'] = precision_score(y_orig, y_pred, average=None)
  resultados['recall'] = recall_score(y_orig, y_pred,average=None)
  resultados['f1'] = f1_score(y_orig, y_pred,average='weighted')
  resultados['kappa'] = cohen_kappa_score(y_orig, y_pred)

  return resultados

filter_sizes = [1,2,3,5,7] #default[1,2,3,5]
num_filters = 36 #default36
batch_size = 128
# model,modelnoactiv = textCNNwithoutembedding(height=newdata.shape[1],width=newdata.shape[2],classes=5,activ='softmax',filter_sizes=filter_sizes,num_filters=num_filters)
model,modelnoactiv = textCNNwithoutembedding(height=1939,width=100,classes=5,activ='softmax',filter_sizes=filter_sizes,num_filters=num_filters)

# load best weights 

best_weights_path = 'best_weights/'
# namefile = drive_dir_weights + 'Round2Fold3_model.weights.best.hdf5' #best accuracy for training on LCT
# namefile = drive_dir_weights + 'Round5Fold1_model.weights.best.hdf5'  #best kappa for training on LCT 
# namefile = best_weights_path + 'ex3/' + 'Round5Fold1_model.weights.best.hdf5'  #best kappa for training on LCT 
# namefile = drive_dir + 'model_weight/61%experiment2_model.weights.best.hdf5' #best weights provided by Leonardo
# namefile = drive_dir_weights + 'Round6Fold1_model.weights.best.hdf5'  #best kappa and accuracy for training on both sets
# namefile = drive_dir_weights + 'Round3Fold9_model.weights.best.hdf5'  #best kappa and accuracy for training on software course
# namefile = best_weights_path + 'ex5/ex5_tsoft/' + 'Round4Fold4_model.weights.best.hdf5'  #best kappa and accuracy for training on both but test on soft (0.1)

namefile = best_weights_path + 'ex5/ex5_tboth/' + 'Round6Fold1_model.weights.best.hdf5'  #best kappa and accuracy for training and testing (0.1) on both

model.load_weights(namefile)

# # load validation set
# # 0.1 (1747) test ratio
# x_valid = np.load("validation/x_validation.npy")
# y_valid = np.load("validation/y_validation.npy")
# index_valid = np.load("validation/index_validation.npy")

# print("validation data loaded...")
# print(x_valid.shape)
# print(len(y_valid))
# # indices = np.arange(len(newdata))
# # # 0.25 test ratio
# # x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(newdata, classes, indices, stratify=classes, random_state = 1 ,test_size=0.25)
# # 0.1 (1747) test ratio

# # y_pred = [np.argmax(lista) for lista in model.predict(x_test)]
# # # resultados = calculatemetrics(y_test,y_pred)
# # # print(resultados)
# # # y_pred = [np.argmax(lista) for lista in model.predict(newdata)]
# # resultados = calculatemetrics(classes,y_pred)
# y_pred = [np.argmax(lista) for lista in model.predict(x_valid)]
# resultados = calculatemetrics(y_valid,y_pred)
# print(resultados)

# # confusion matrix 
# cm = confusion_matrix(y_pred,y_valid)
# print(cm)
# print(y_valid)
# print(index_valid)

########### ex5 tsoft the testing set is from the software set
# get the x_train and y_train from deducting the validation data from the newdata (entire set)
# x_train_1 = np.delete(course1_set_new, index_valid, axis=0)
# x_train = np.concatenate((x_train_1, course2_set_new), axis = 0)

# # new list to append the ndarry not in the x_valid 
# y_train = []
# index_train = []

# for i in range(len(course1_class)):
#   if i not in index_valid:
#     y_train.append(course1_class[i])
#     index_train.append(i)
    
# print(len(y_train))

# for j in range(len(course2_class)):
#   y_train.append(course2_class[j])
#   index_train.append(j + len(course1_class))


# print(x_train.shape)
# print(len(y_train))
# print(len(index_train))

# ########### ex5 tlct the testing set is from the software set
# # load the x  y_valid for the testing work after the cv selection
# x_valid = np.load("validation/x_validation_ex5lct.npy")
# y_valid = np.load("validation/y_validation_ex5lct.npy")
# index_valid = np.load("validation/index_validation_ex5lct.npy") # index_valid is the index in the individual course2 (lct ) #should add 1747 in the combination set (0-1478)

# print("validation data " + str(len(x_valid)) + " loaded.")

# # load the training data for cv loop
# cv_set = np.load("validation/cv_set_ex5lct.npy")
# cv_classes = np.load("validation/cv_classes_ex5lct.npy")
# print("training (CV) data " + str(len(cv_set)) + " loaded.")

# #get the index of the cv_set 
# index_cv = []

# for i in range(len(course1_class)):
#   index_cv.append(i)

# for j in range(len(course2_class)):
#   if j not in index_valid:
#     index_cv.append(j+len(course1_class))

# print("index_cv generated.")
# print(index_cv)

# #change the index of the validation set
# [item+ len(course1_class) for item in index_valid]
  
# print("index_valid changed.")
# print(index_valid)

# y_pred = [np.argmax(lista) for lista in model.predict(x_valid)]
# resultados = calculatemetrics(y_valid,y_pred)
# print(resultados)

########### ex5 tboth the testing set is from both sets
x_valid = np.load("validation/x_validation_ex5both.npy")
y_valid = np.load("validation/y_validation_ex5both.npy")

y_pred = [np.argmax(lista) for lista in model.predict(x_valid)]
resultados = calculatemetrics(y_valid,y_pred)
print(resultados)

# confusion matrix 
cm = confusion_matrix(y_pred,y_valid)
print(cm)
print(y_valid)

'''

########### ex5 tboth the testing set is from the software set


########################## explainable AI visualisation on testing dataset #########################
# Here i create the analyzer, not every analyzer available in this framework works
# I'm going to look again the list and send to you, the ones who works
gradientanalyzer = create_analyzer('smoothgrad',modelnoactiv)

indices = np.arange(len(newdata))
# i will let this code here just in case you wanna do a quick test in the XAI
# x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(newdata, classes, indices, stratify=classes, random_state = 1 ,test_size=0.25)

# you have to pass the X data, Y data, the model, the modelwithout the activation
# and the gradientanalyzer created
# about the X data and the Y data, it can be the entire dataset, in my experiments i only pass
# the training data, but its not obligatory.

analisesgradient = generateanalisys(x_train,y_train,modelnoactiv,model,gradientanalyzer)

scores = [analise[(len(analise) - len(word)) : len(analise) ] for analise,word in zip(analisesgradient,np.array(preprocessed)[index_train])]

for x in range(50,55):
# Here i get the original class of the example
  classeorig = classes[index_train[x]]

# Here i get the predicted class of the example
  classepredict = np.argmax(model.predict(np.expand_dims(newdata[index_train[x]],axis = 0)))

# Here is the function to generate the text_heatmap
  plot_text_heatmap(preprocessed[index_train[x]], scores[x],number = x, title=str(x) + " Class: " + str(classeorig) + " Predicted class: " + str(classepredict))

'''

