#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import CuDNNLSTM, TimeDistributed, Bidirectional
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

L_S = 20    # nb of intervals between similarity samplings for simulations
S_range = np.linspace(0,1,L_S+1)   # Similarity between founders

L_V = 10    # nb of intervals between variance samplings for simulations
V_range = np.linspace(0,0.5,L_V+1)   # standard deviation of the cross-similarity between founders

linkage = np.array([1/(p-1) for k in range(p)])   # uniform probability of recombination

F_haplotypes = []
d_haplotypes = []
F_genotypes = []
d_genotypes = []
for S in S_range[::2] :
    for V in V_range :
        with open("Simulations/Simulations_haplotypes_n{0}_p{1}_G{2}_N{3}_I{4}_S{5:.2f}_V{6:.2f}.pkl".format(n,p,G,N,I,S,V),"rb") as handler:
            haplo = pickle.load(handler)
            F_haplotypes.append(np.array(haplo[:n]))
            d_haplotypes.append(np.array(haplo[n:]))
        with open("Simulations/Simulations_genotypes_n{0}_p{1}_G{2}_N{3}_I{4}_S{5:.2f}_V{6:.2f}.pkl".format(n,p,G,N,I,S,V),"rb") as handler:
            haplo = pickle.load(handler)
            F_genotypes.append(np.array(haplo[:n]))
            d_genotypes.append(np.array(haplo[n:]))

print(len(d_genotypes), F_haplotypes[0].shape, d_haplotypes[0].shape, F_genotypes[0].shape, d_genotypes[0].shape, linkage.shape)

##
with open('Similarity3_model_history/Similarity3_model_history_S0.5_V0.0.pkl', 'rb') as handler:
    history = pickle.load(handler)
history["val_acc"]
##
max_acc = np.zeros((L_S//2+1, L_V+1))

for i in range(0,L_S+1,2) :
    S = S_range[i]
    print ("S : {0}/{1}".format(int(S*L_S),int(S_range[-1]*L_S)))
    for j in range(L_V+1) :
        V = V_range[j]
        print ("V : {0}/{1}".format(int(V*L_V*2),int(V_range[-1]*L_V*2)))
        with open('Similarity3_model_history/Similarity3_model_history_S{}_V{}.pkl'.format(round(S,2),round(V,2)), 'rb') as handler:
            history = pickle.load(handler)
        max_acc[i//2,j] = max(history.history["val_acc"])

with open('Maximum_accuracy.pkl', 'wb') as handler:
    pickle.dump(max_acc, handler)