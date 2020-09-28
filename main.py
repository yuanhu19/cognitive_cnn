# imports
import os
import pickle
import time
import imp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
import json

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
originaldata = pd.read_excel(datapath1)

# # use pandas to load the data from the LCT MOOC
# originaldata = pd.read_excel(datapath2)

# #load both datasets
# data1 = pd.read_excel(datapath1)[['phaseId','postBody']]
# data1['courseType'] = 1             #software engineering
# data2 = pd.read_excel(datapath2)[['phaseId','postBody']]
# data2['courseType'] = 2             #LCT MOOC

# #combine two dfs / shuffle the rows / reindex 3226 messages in total
# originaldata = pd.concat([data1,data2])
# originaldata = originaldata.sample(frac=1, random_state=1)
# originaldata.reset_index(drop=True, inplace=True)

# getting the class
classes = originaldata['phaseId'].tolist()

# changing the format of the class to categorical
encodedclass = to_categorical(classes)
# IDs = originaldata['courseID'].tolist() # courseID in the software dataset
# IDs = originaldata['postSectionId'].tolist() # sectionID in the LCT MOOC dataset
# getting the texts
texts = originaldata['postBody'].tolist()

# print(len(classes))
print(len(encodedclass))
print(originaldata)


########################## pre processing #########################
# Loading the glove into a word2vec gensim object
word2vec_filenametxt = 'glovefiles/' + 'glove.6B.100d.txt'
word2vec_filename = 'glovefiles/' + 'glove.6B.100d.txt.word2vec'
# word2vec_filename = 'glovefiles/' + 'glove.6B.100d.w2vformat.txt'
modelglove = KeyedVectors.load_word2vec_format(word2vec_filename, binary=False)
print('glove loaded.')
print('text converting...')

newdata, data, preprocessed, heigth ,width, glovemodel = generate_cnndata(texts, model = modelglove,  path = word2vec_filename)
print('preprocessed finished.')


'''
########################## training #########################

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


for seedkfold in [1,2,3,4,5,6]:
  
  kfold = StratifiedKFold(sizekfold,True,seedkfold)
  foldnumber = 1

  for train_index,test_index in kfold.split(newdata,classes):

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
  
    x_train, x_test = np.array(newdata)[train_index],np.array(newdata)[test_index]
    y_train, y_test, y_tousemetrics = np.array(encodedclass)[train_index], np.array(encodedclass)[test_index], np.array(classes)[test_index]

    model,modelnoactiv = textCNNwithoutembedding(height=newdata.shape[1],width=newdata.shape[2],classes=nclasses,activ=activ,filter_sizes=filter_sizes,num_filters=num_filters)

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

#results of lct datasets
df_result.to_csv( "result_soft.csv", index = True, header=True)

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
model,modelnoactiv = textCNNwithoutembedding(height=newdata.shape[1],width=newdata.shape[2],classes=5,activ='softmax',filter_sizes=filter_sizes,num_filters=num_filters)

best_weights_path = 'best_weights/'
# namefile = drive_dir_weights + 'Round2Fold3_model.weights.best.hdf5' #best accuracy for training on LCT
# namefile = drive_dir_weights + 'Round5Fold1_model.weights.best.hdf5'  #best kappa for training on LCT 
namefile = best_weights_path + 'ex3/' + 'Round5Fold1_model.weights.best.hdf5'  #best kappa for training on LCT 
#namefile = drive_dir + 'model_weight/61%experiment2_model.weights.best.hdf5' #best weights provided by Leonardo
# namefile = drive_dir_weights + 'Round6Fold1_model.weights.best.hdf5'  #best kappa and accuracy for training on both sets
# namefile = drive_dir_weights + 'Round3Fold9_model.weights.best.hdf5'  #best kappa and accuracy for training on software course

model.load_weights(namefile)

# indices = np.arange(len(newdata))
# # 0.25 test ratio
# x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(newdata, classes, indices, stratify=classes, random_state = 1 ,test_size=0.25)

# y_pred = [np.argmax(lista) for lista in model.predict(x_test)]
# resultados = calculatemetrics(y_test,y_pred)
# print(resultados)
y_pred = [np.argmax(lista) for lista in model.predict(newdata)]
resultados = calculatemetrics(classes,y_pred)
print(resultados)


'''
########################## explainable AI visualisation on testing dataset #########################
# Here i create the analyzer, not every analyzer available in this framework works
# I'm going to look again the list and send to you, the ones who works
gradientanalyzer = create_analyzer('smoothgrad',modelnoactiv)

indices = np.arange(len(newdata))
# i will let this code here just in case you wanna do a quick test in the XAI
x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(newdata, classes, indices, stratify=classes, random_state = 1 ,test_size=0.25)

# you have to pass the X data, Y data, the model, the modelwithout the activation
# and the gradientanalyzer created
# about the X data and the Y data, it can be the entire dataset, in my experiments i only pass
# the training data, but its not obligatory.
analisesgradient = generateanalisys(x_train,y_train,modelnoactiv,model,gradientanalyzer)

# scores = [analise[(len(analise) - len(word)) : len(analise) ] for analise,word in zip(analisesgradient,np.array(preprocessed)[index_train])]
scores = [analise[(len(analise) - len(word)) : len(analise) ] for analise,word in zip(analisesgradient,np.array(preprocessed)[index_test])]

for x in range(50,81):
# X is the value of the position of the example you want to get in your dataset to make the xai representation 
  # x = 2

# Here i get the original class of the example
  # classeorig = classes[index_train[x]]

# Here i get the predicted class of the example
  # classepredict = np.argmax(model.predict(np.expand_dims(newdata[index_train[x]],axis = 0)))

# Here is the function to generate the text_heatmap
  # plot_text_heatmap(preprocessed[index_train[x]], scores[x], title=str(x) + " Class: " + str(classeorig) + " Predicted class: " + str(classepredict))

  classeorig = classes[index_test[x]]

# Here i get the predicted class of the example
  classepredict = np.argmax(model.predict(np.expand_dims(newdata[index_test[x]],axis = 0)))

# Here is the function to generate the text_heatmap
  plot_text_heatmap(preprocessed[index_test[x]], scores[x], title=str(x) + " Class: " + str(classeorig) + " Predicted class: " + str(classepredict))
  
'''