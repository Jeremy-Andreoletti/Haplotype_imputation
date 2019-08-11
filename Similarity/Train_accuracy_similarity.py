#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, TimeDistributed, Bidirectional
from keras.layers import Conv1D, Conv2D, MaxPooling1D, AveragePooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import np_utils, plot_model
import pickle
import os

## Import data and separate 3/4 of data for training and 1/4 for testing.

if os.getcwd() == '/Users/jeje' :
    os.chdir("/Users/jeje/Documents/Scolaire/ENS/Immersion_experimentale/Similarity/")

n = 16    # number of founders
p = 1000  # nb of SNPs
G = 20    # nb of generations
N = 100   # nb of recombining individuals in a generation after sampling
I = 100   # nb of itérations

L = 20    # nb of intervals between similarity samplings for simulations
S_range = np.linspace(0,1,L+1)   # Similarity between founders

linkage = np.array([1/(p-1) for k in range(p)])   # Uniform probability of recombination

haplotypes = []
F_haplotypes = []
d_haplotypes = []
for S in S_range :
    with open("Simulations/Simulations_haplotypes_n{0}_p{1}_G{2}_N{3}_I{4}_S{5:.2f}.pkl".format(n,p,G,N,I,S),"rb") as handler:
        haplo = pickle.load(handler)
        F_haplotypes.append(np.array(haplo[:n]))
        d_haplotypes.append(np.array(haplo[n:]))
        
genotypes = []
F_genotypes = []
d_genotypes = []
for S in S_range :
    with open("Simulations/Simulations_genotypes_n{0}_p{1}_G{2}_N{3}_I{4}_S{5:.2f}.pkl".format(n,p,G,N,I,S),"rb") as handler:
        haplo = pickle.load(handler)
        F_genotypes.append(np.array(haplo[:n]))
        d_genotypes.append(np.array(haplo[n:]))

print(len(d_genotypes), F_haplotypes[0].shape, d_haplotypes[0].shape, F_genotypes[0].shape, d_genotypes[0].shape, linkage.shape)

## Shape inputs and outputs

def chromosome_alternate_inputs_5 (array, F_geno, linkage, n):
    A = np.zeros((array.shape[0],array.shape[1],3*n+1))
    for k in range(len(array)) :
        individual = array[k]
        a = individual[:, np.newaxis]
        for l in range(len(F_geno)) :
            a=np.concatenate([a,F_geno[l:l+1].T, linkage[:, np.newaxis], individual[:, np.newaxis]], axis = 1)
        A[k]=a
    return A

def chromosome_segments_output (array):
    A = np.zeros((len(array),len(array[0])//2,16))
    for k in range(len(array)) :
        individual = array[k]
        A[k] = np.array([np_utils.to_categorical(segment[1]-1,16) for segment in np.reshape(individual[:len(individual)//2*2],(-1,2))])
    return A

def prepare_data (d_geno, d_haplo, F_geno, linkage, n) :
    xtrain = d_geno[:9*I*N//10]
    ytrain = d_haplo[:9*I*N//10]
    xtest = d_geno[9*I*N//10:]
    ytest = d_haplo[9*I*N//10:]
    
    Xtrain = chromosome_alternate_inputs_5(xtrain, F_geno, linkage, n)
    Ytrain = chromosome_segments_output(ytrain)
    Xtest = chromosome_alternate_inputs_5(xtest, F_geno, linkage, n)
    Ytest = chromosome_segments_output(ytest)
    
    return Xtrain, Ytrain, Xtest, Ytest

## Define the model

K.clear_session()

model = Sequential()
model.add(Conv1D(49, kernel_size=2,
                 activation='relu',
                 input_shape=(p,3*n+1), padding='same'))
model.add(Conv1D(30, kernel_size=2, activation='relu', padding='same'))
model.add(AveragePooling1D(pool_size=2,strides=2))
model.add(Dropout(0.25))
model.add(Conv1D(16, kernel_size=2, activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(TimeDistributed(Dense(16, activation='relu')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(16, activation='relu')))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(8, return_sequences=True)))
model.add(TimeDistributed(Dense(n, activation='softmax')))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
model.save_weights("Similarity2_model_initial_weights.hdf5")
print(model.summary())

## Train the model

def run_model (model, S, Xtrain, Ytrain, Xtest, Ytest, epochs):
    model.load_weights("Similarity2_model_initial_weights.hdf5")
    filepath="Similarity2_model_weights-improvement-"+"S_{}".format(round(S,2))+"-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(Xtrain, Ytrain,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        verbose=2,
                        validation_data=(Xtest, Ytest))
    return history

epochs = 100

for k in range(L+1) :
    S = S_range[k]
    d_geno = d_genotypes[k]
    F_geno = F_genotypes[k]
    d_haplo = d_haplotypes[k]
    Xtrain, Ytrain, Xtest, Ytest = prepare_data (d_geno, d_haplo, F_geno, linkage, n)
    history = run_model (model, S, Xtrain, Ytrain, Xtest, Ytest, epochs)
    with open('Similarity2_model_history_S{}.pkl'.format(round(S,2)), 'wb') as handler:
        pickle.dump(history, handler)