#!/usr/bin/env python

import nngp
import numpy as np
import sys,os


# Input files
EP='METAB'
ftrain=EP+'_training.csv'
ftest=EP+'_test.csv'

# output directory
OUT='TEST'


# INITIALIZE Model
###################################
mymod=nngp.NNGaussianProcess(ifname=ftrain,modelpath=OUT,kneighbors=100,antrees=50,chunk=500)
mp=mymod.modelpath


# VALIDATE
###################################
nfit=5000 # max number of training compounds to fit kernel parameters
vf=0.25 # fraction of 'ftrain' to validate on

_,R2valc=mymod.train_mod(nfit,valmethod='chunk',valfrac=vf)
print('time-split validation R2 = ',R2valc)

_,R2valcr=mymod.train_mod(nfit,valmethod='chunk',valfrac=vf,randomsplit=True)
print('random-split validation R2 = ',R2valcr)


# TRAIN Model
###################################
mymod.train_mod(nfit)

#SAVE Trained model to modelpath
###################################
mymod.save_mod()


# TEST on separate input file 'ftest'
###################################
preds,R2test=mymod.predict(ifname=ftest,save=True,outfile=EP+'_pred.csv')
print('prediction R2 = ',R2test)


# UPDATE Methods
###################################
mymod.load_mod(ifname=mp) # reload model
preds,R2upc4,numup=mymod.update_mod(ifname=ftest,valmethod='chunk 4')
print('update in 4 chunks, R2 = ',R2upc4)

mymod.load_mod(ifname=mp) # reload model
preds,R2upcontemp,numup=mymod.update_mod(ifname=ftest,valmethod='contemp week')
print('contemporaneous update, R2 = ',R2upcontemp)

mymod.load_mod(ifname=mp) # reload model
preds,R2upcweek,nupcweek=mymod.update_mod(ifname=ftest,valmethod='chunk week')
print('weekly update, R2 = ',R2upcweek)


#SAVE Updated Model in {modelpath}/updated/
###################################
mymod.save_mod(modelpath=mp+'/updated/')
    
    
