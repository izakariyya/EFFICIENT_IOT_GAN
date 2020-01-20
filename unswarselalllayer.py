# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:44:10 2019

@author: 1804499
"""

from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


from hyperas import optim
from hyperas.distributions import choice, uniform

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    benign = np.loadtxt("UNSW_NB15_Train.csv", delimiter = ",")
    benscan = np.loadtxt("UNSW_NB15_Test.csv", delimiter = ",")
    alldata = np.concatenate((benign, benscan))
    j = len(benscan[0])
    data = alldata[:, 1:j]
    data = PCA(0.99).fit_transform(data)
    benlabel = alldata[:, 0]
    bendata = (data - data.min()) / (data.max() - data.min())
    bendata, benmir, benlabel, benslabel = train_test_split(bendata, benlabel, test_size = 0.2)
    return bendata, benlabel, benmir, benslabel

def new_model(bendata, benlabel, benmir, benslabel):
    
    n = len(bendata[0])
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense({{choice([32,64, 128, 256, 512, 1024])}}, input_shape=(n,)))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    #model.add({{choice([Dropout(0.2)])}})
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([32,64, 128, 256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    #model.add({{choice([Dropout(0.2)])}})
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if {{choice(['two', 'three'])}} == 'three':
        model.add(Dense({{choice([32,64, 128, 256, 512, 1024])}}))
        model.add(Activation({{choice(['relu', 'sigmoid'])}}))
        #model.add({{choice([Dropout(0.2)])}})
        model.add(Dropout({{uniform(0, 1)}}))
        
        # We can also choose between complete sets of layers

        #model.add({{choice([Dropout(0.5), Activation('sigmoid')])}})
        #model.add(Activation('relu'))
        
    model.add(Dense(1))
    model.add(Activation({{choice(['sigmoid', 'softmax', 'linear'])}}))
    
    adam = Adam(lr={{choice([10**-4, 10**-3, 10**-2, 10**-1])}})
    rmsprop = RMSprop(lr={{choice([10**-4, 10**-3, 10**-2, 10**-1])}})
    sgd = SGD(lr={{choice([10**-4, 10**-3, 10**-2, 10**-1])}})
    
    chosen_par = {{choice(['adam', 'rmsprop', 'sgd'])}}
    if chosen_par == 'adam':
        optm = adam
    elif chosen_par == 'rmsprop':
        optm = rmsprop
    else:
        optm = sgd
    
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=optm)

    
    result = model.fit(bendata, benlabel,
              batch_size={{choice([16, 32, 64, 128, 256])}},
              #batch_size = 32,
              epochs ={{choice([2,4,10,15,20])}},
              #epochs=2,
              verbose=2,
              validation_split=0.1)
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best test accuracy of epoch:',   validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
   
#model execution

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=new_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)





