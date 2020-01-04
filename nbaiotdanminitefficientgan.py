# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:22:29 2019

@author: 1804499
"""
#importing standard modules
import os
import csv
import time
from collections import defaultdict
import numpy as np
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras.optimizers import Adam, RMSProp, SGD
from keras.utils.generic_utils import Progbar
from sklearn.model_selection import train_test_split
from memory_profiler import profile

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

#Function to load, normalize and splits datasets in csv format
def Load_Data():
    benign = np.loadtxt("benign_train.csv", delimiter = ",")
    benscan = np.loadtxt("ben_mir_gaf.csv", delimiter = ",")
    alldata = np.concatenate((benign, benscan))
    j = len(benscan[0])
    data = alldata[:, 1:j] 
    benlabel = alldata[:, 0]
    bendata = (data - data.min()) / (data.max() - data.min())
    bendata, benmir, benlabel, benslabel = train_test_split(bendata, benlabel, test_size = 0.2)
    return bendata, benmir, benlabel, benslabel
    
# Functions for returning optimizer with an optimal learning rate
    
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

#generator functional model definition
    
def gen_model(optimizer, n):
    
    inputs = Input((n,))
    l1 = Dense(input_dim = n , units = 64, kernel_initializer=initializers.glorot_uniform())(inputs)
    l1 = BatchNormalization()(l1)
    l1 = Activation('relu')(l1)
    
    l2 = Dense(128)(l1)
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
    
    l2 = Dense(128)(l1)
    l2 = LeakyReLU(0.1)(l2)
    l2 = Dropout(0.2)(l2)
    
    l3 = Dense(128)(l2)
    l3 = LeakyReLU(0.1)(l3)
    l3 = Dropout(0.2)(l3)
    
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

# training start time    
start_train_time = time.time()

#training memory 

precision = 10

fp = open('BAIOT_Mem_Train.log', 'w+')

@profile(precision=precision, stream=fp)



def train(BATCH_SIZE, X_train, Xlabel, n):
    #Building model
    dsize = len(X_train[0])
    # optimizer return with the learnig rate
    adam = get_optimizer()
    
    generator = gen_model(adam, n)
    discriminator = dis_model(adam, n)

    gan = define_gan(discriminator,generator, adam, n)

    generator.compile(loss='binary_crossentropy', optimizer= adam)
    
    gan.compile(loss='binary_crossentropy', optimizer= adam, metrics =['accuracy'])
    discriminator.trainable = True

    discriminator.compile(loss='binary_crossentropy', optimizer= adam, metrics =['accuracy'])
    
    train_history = defaultdict(list)
    
    
    for epoch in range(5):
        print ("Epoch is", epoch)
        n_iter = int(np.ceil(X_train.shape[0]/ float(BATCH_SIZE)))
        progress_bar = Progbar(target=n_iter)
        
        epoch_dloss = []
        epoch_gloss = []
        
        for index in range(n_iter):
            
            # load real data & real_label
            real_data =  X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]  
            real_label = Xlabel[index*BATCH_SIZE:(index+1)*BATCH_SIZE]  
                      
            # create random noise -> U(0,1) n dimension latent vectors
            noise = np.random.normal(0, 1, (len(real_data), dsize))
             
            #generate fake data
            gen_data = generator.predict(noise, verbose=0)
            
            
            X = np.concatenate((real_data, gen_data))
            
            #attach labels for training the discriminator
            ylbl = np.array([0]*len(real_data))
            ylabel = np.concatenate((real_label, ylbl))
            
            
            # training discriminator
            
            epoch_dloss.append(discriminator.train_on_batch(X, ylabel))
            
            dis_loss = np.mean(np.array(epoch_dloss))
            
            #create another set of random noise for the generator
            gan_noise = np.random.normal(0, 1, (2*len(real_data), dsize))
            
            gantrick = np.concatenate((real_label, np.ones(len(real_data))))
            
            # training generator
            discriminator.trainable = False
            
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



# Get the preprocessed train and testing data

trainddos, testddos, trainddoslbl, testddoslbl =  Load_Data()

n = len(trainddos[0])

#Instantiation of the GAN training with the train samples data

d, g, gn = train(16, trainddos, trainddoslbl, n)

#End of trainig
end_train_time = time.time()

print(end_train_time - start_train_time)

#Starting Test Time

start_test_time = time.time()

precision = 10

#Testing memory

fp = open('BAIOT_Mem_Test.log', 'w+')

@profile(precision=precision, stream=fp)

def generate_result(d, g, gn, test_data, test_label, dsize):
    test_history = defaultdict(list)   
    dsize = len(test_data[0])
    num_test = test_data.shape[0]
    #generating fake data
    noise = np.random.normal(0,1, (num_test, dsize))
    generated_data = g.predict(noise, verbose=False)
    new_data = np.concatenate((test_data, generated_data))
    y = np.array([0]*num_test)
    #attaching label for testing
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
    # writing the predicted labels
    res = d.predict(new_data)
    with open('benign_scan_prediction_result.csv', 'w', newline ='') as f:
        writer = csv.writer(f)
        for x in res:
            writer.writerow(x)
    return dis_tloss*100


result = generate_result(d, g, gn, testddos, testddoslbl, n)
#outputing loss and accuracy
print(result[0], result[1])

end_test_time = time.time()
#outputing the ending testing time
print(end_test_time -start_test_time )  
    
