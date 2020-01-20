# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:40:38 2019

@author: 1804499
"""
import os
import csv
import time
import math
from collections import defaultdict
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Reshape, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from matplotlib import pyplot
from numpy.random import rand
from numpy.random import randn
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers
import tensorflow as tf
from tqdm import tqdm
from keras.utils.generic_utils import Progbar
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from memory_profiler import profile
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from sklearn_pandas import CategoricalImputer
import pandas as pd

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

def data():
   
    benign = np.loadtxt("UNSW_NB15_Train.csv", delimiter = ",")
    benscan = np.loadtxt("UNSW_NB15_Test.csv", delimiter = ",")
    alldata = np.concatenate((benign, benscan))
    j = len(benscan[0])
    data = alldata[:, 1:j]
    data = PCA(0.99).fit_transform(data)
    benlabel = alldata[:, 0]
    bendata = (data - data.min()) / (data.max() - data.min())
    bendata, benmir, benlabel, benslabel = train_test_split(bendata, benlabel, test_size = 0.2)
    return bendata, benmir, benlabel, benslabel

def get_optimizer():
    #return RMSprop(lr=0.0001)
    #return RMSprop(lr=0.0002)
    #return RMSprop(lr=0.0004)
    return Adam(lr=0.0001)
    

#generator functional model definition
    
def gen_model(optimizer, n):
    
    inputs = Input((n,))
    l1 = Dense(input_dim = n , units = 64, kernel_initializer=initializers.glorot_uniform())(inputs)
    l1 = BatchNormalization()(l1)
    l1 = Activation('relu')(l1)
    
    l2 = Dense(512)(l1)
    l2 = BatchNormalization()(l2)
    l2 = Activation('relu')(l2)
    
    l3 = Dense(n)(l2)
    outputs = Activation('relu')(l3)
    
    gmodel = Model(inputs = [inputs], outputs = [outputs])
    #gmodel.compile(loss='binary_crossentropy', optimizer= optimizer)
    #print(gmodel.summary)
    return gmodel
#define the discriminator functional model
    
def dis_model(optimizer, n):
    inputs = Input((n,))
    l1 = Dense(input_dim = n, units = 64, kernel_initializer=initializers.glorot_uniform())(inputs)
    l1 = LeakyReLU(0.1)(l1)
    l1 = Dropout(0.2)(l1)
    
    l2 = Dense(512)(l1)
    l2 = LeakyReLU(0.1)(l2)
    l2 = Dropout(0.3)(l2)
    
    #l3 = Flatten()(l2)
    l3 = Dense(1024)(l2)
    l3 = LeakyReLU(0.1)(l3)
    l3 = Dropout(0.4)(l3)
    
    l4 = Dense(1)(l3)
    outputs = Activation('sigmoid')(l4)
    
    dmodel = Model(inputs = [inputs], outputs = [outputs])
    #dmodel.trainable = True
    #dmodel.compile(loss='binary_crossentropy', optimizer= optimizer, metrics =['accuracy'])
    #print(dmodel.summary)
    return dmodel

def define_gan(discriminator, generator, optimizer, n):
    discriminator.trainable = False
    gan_input = Input(shape=(n,))
    # the output of the generator (
    x = generator(gan_input)
    #x = x.reshape(data_size, dsize)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    #gan.compile(loss='binary_crossentropy', optimizer= optimizer, metrics =['accuracy'])
    #print(gan.summary)
    return gan
    
start_train_time = time.time()

precision = 10

fp = open('UNSW_PCA_Mem_Train.log', 'w+')

@profile(precision=precision, stream=fp)



def train(BATCH_SIZE, X_train, Xlabel, n):
    #Building model
    dsize = len(X_train[0])
    adam = get_optimizer()
    #gen_opz = RMSprop(lr = 0.0002)
    #dis_opz = RMSprop(lr = 0.0004)
    #generator = gen_model(gen_opz, n)
    generator = gen_model(adam, n)
    #discriminator = dis_model(dis_opz, n)
    discriminator = dis_model(adam, n)
    #gan = define_gan(discriminator,generator, gen_opz, n)
    gan = define_gan(discriminator,generator, adam, n)
    #generator.compile(loss='binary_crossentropy', optimizer= gen_opz)
    generator.compile(loss='binary_crossentropy', optimizer= adam)
    #gan.compile(loss='binary_crossentropy', optimizer= gen_opz, metrics =['accuracy'])
    gan.compile(loss='binary_crossentropy', optimizer= adam, metrics =['accuracy'])
    discriminator.trainable = True
    #discriminator.compile(loss='binary_crossentropy', optimizer= dis_opz, metrics =['accuracy'])
    discriminator.compile(loss='binary_crossentropy', optimizer= adam, metrics =['accuracy'])
    
    train_history = defaultdict(list)
    
    
    for epoch in range(2):
        print ("Epoch is", epoch)
        n_iter = int(np.ceil(X_train.shape[0]/ float(BATCH_SIZE)))
        progress_bar = Progbar(target=n_iter)
        
        epoch_dloss = []
        epoch_gloss = []
        
        for index in range(n_iter):
            
            # load real data & real_label
            real_data =  X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]  
            real_label = Xlabel[index*BATCH_SIZE:(index+1)*BATCH_SIZE]  
                      
            # create random noise -> U(0,1) 115 latent vectors
            noise = np.random.normal(0, 1, (len(real_data), dsize))
            
            #labels sampling
            #generate random label
            #sample_label = np.random.randint(0,2, (len(real_data)))
            
            #generate fake data
            gen_data = generator.predict(noise, verbose=0)
            
            #attach labels for training the discriminator
            X = np.concatenate((real_data, gen_data))
            #ylabel = np.array([1]*len(real_data)+[0]*len(real_data))
            
            ylbl = np.array([0]*len(real_data))
            ylabel = np.concatenate((real_label, ylbl))
            
            #Y = np.concatenate((real_label, sample_label), axis = 0)
            
            # attach label for training discriminator
            #soft_zero, soft_one = 0, 0.95
            #y = np.array([soft_one]*len(real_data)+[soft_zero]*len(real_data))
            
            
            # training discriminator
            
            #discriminator.train_on_batch(X, Y)
            epoch_dloss.append(discriminator.train_on_batch(X, ylabel))
            #epoch_dloss.append(discriminator.train_on_batch(X, Y))
            dis_loss = np.mean(np.array(epoch_dloss))
            
            #create another set of random noise for the generator
            gan_noise = np.random.normal(0, 1, (2*len(real_data), dsize))
            #gan_sample_label = np.random.randint(0, 2, 2*len(real_data))
            
            #trick = np.ones(2*len(real_data)) *soft_one
            #gantrick = np.ones(2*len(real_data)) 
            
            gantrick = np.concatenate((real_label, np.ones(len(real_data))))
            
            # training generator
            discriminator.trainable = False
            #epoch_gloss.append(gan.train_on_batch(gan_noise, gan_sample_label))
            epoch_gloss.append(gan.train_on_batch(gan_noise, gantrick))
            gen_loss = np.mean(np.array(epoch_gloss))
            discriminator.trainable = True
            
            train_history['generator'].append(gen_loss)
            train_history['discriminator'].append(dis_loss)

            
            progress_bar.update(index + 1)
        #print ('')
        
        # save weights for each epoch
        #generator.save_weights('weights/generator.h5')
        #discriminator.save_weights('weights/discriminator.h5')
        
    return discriminator, generator, gan




trainddos, testddos, trainddoslbl, testddoslbl =  data()

n = len(trainddos[0])

d, g, gn = train(64, trainddos, trainddoslbl, n)

#trainiotpca, testiotpca, trainiotpcalbl, testiotpcalbl =  Load_Data()

end_train_time = time.time()

print(end_train_time - start_train_time)

start_test_time = time.time()

precision = 10

fp = open('UNSW_PCA_Mem_Test.log', 'w+')

@profile(precision=precision, stream=fp)

def generate_result(d, g, gn, test_data, test_label, dsize):
    test_history = defaultdict(list)   
    dsize = len(test_data[0])
    num_test = test_data.shape[0]
    noise = np.random.normal(0,1, (num_test, dsize))
    #sample_label = np.random.randint(0, 2, num_test)
    generated_data = g.predict(noise, verbose=False)
    new_data = np.concatenate((test_data, generated_data))
    #Y = np.array([1]*num_test + [0]*num_test)
    y = np.array([0]*num_test)
    Y = np.concatenate((test_label, y))
    #dicriminator test loss
    dis_tloss = d.evaluate(new_data, Y, verbose = False)
    test_history['discriminator'].append(dis_tloss)
    new_noise =np.random.normal(0,1, (2*num_test, dsize))
    #new_sample_label = np.random.randint(0,2, 2*num_test)
    trick = np.ones(num_test)
    ganl = np.concatenate((test_label, trick))
    #gen_test_loss = gn.evaluate(new_noise, new_sample_label, verbose = False)
    gen_test_loss = gn.evaluate(new_noise, ganl, verbose = False)
    test_history['generator'].append(gen_test_loss)
    #score = d.evaluate(test_data, test_label, verbose = 0)
    res = d.predict(new_data)
    with open('benign_scan_prediction_result.csv', 'w', newline ='') as f:
        writer = csv.writer(f)
        for x in res:
            writer.writerow(x)
    return dis_tloss*100


result = generate_result(d, g, gn, testddos, testddoslbl, n)

print(result[0], result[1])

end_test_time = time.time()

print(end_test_time -start_test_time )  
    
