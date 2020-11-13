#!/bin/env python 

# The MIT License

# Copyright (c) 2020 Merck Sharp & Dohme Corp. a subsidiary of Merck & Co., Inc., Kenilworth, NJ, USA.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Nearest Neighbor Gaussian Process

written by: Anthony DiFranzo and Matthew Tudor

Last modified: Oct 2, 2020
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.stats import linregress
from scipy import sparse
from rdkit.Chem import SanitizeMol, MolFromSmiles, SanitizeFlags
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import RDLogger
import os, sys
import annoy
import pickle
import warnings
import re
import datetime
import random

lg=RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

class NNGaussianProcess:
    def __init__(self,
                 ifname=None,
                 x=None,y=None,ids=None,dates=None,
                 kneighbors=100,antrees=50,chunk=500,
                 nbits=1024,radius=3,
                 transforms='scale',
                 modelpath=None
                 ):

        # initialize model
        self.x=None
        self.y=None
        self.ids=None
        self.dates=None
        self.k=None
        self.an=None

        self.kneighbors=kneighbors
        self.antrees=antrees
        self.chunk=chunk
        self.nbits=nbits
        self.radius=radius

        # read and process data
        if ifname is not None:
            # if file with data is given
            if os.path.isfile(ifname):
                self.x,self.y,self.ids,self.dates=self.read_input(ifname)
                self.transform_y(transforms)
                self.x=norm_x(self.x) 
                if modelpath is None:
                    now=datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d')
                    modelpath=os.path.splitext(os.path.basename(ifname))[0]+'_'+now
            # if path to previously saved model is given
            elif os.path.isdir(ifname):
                self.load_mod(ifname)
                if modelpath is None: modelpath=ifname
            else: sys.exit(ifname+' does not exist.\n'
                           +'Enter a valid input data file or a directory containing a previously trained model.')

        # if processed data or incomplete/no data is given
        elif x is not None and y is not None:
            if isinstance(x[0],str):
                xtmp=np.zeros((len(x),self.nbits))
                for i,smi in enumerate(x):
                    xtmp[i]=get_ECFP(smi,nbits=self.nbits,radius=self.radius)
                x=xtmp
            if x.shape[1]!=self.nbits: sys.exit("'x' is invalid.\n"
                           +"It must be a list of SMILES strings or a list of fingerprints, with fingerprint length matching 'nbits'")
            if not sparse.issparse(x): x=sparse.csr_matrix(x,dtype=np.float32)
            x=norm_x(x)

            if not isinstance(y,np.ndarray): y=np.array(y)

            if ids is None: ids=[str(i) for i in range(len(y))]
            elif not isinstance(ids,list): ids=list(ids)

            if dates is None: dates=list(range(len(y)))
            elif not isinstance(dates,list): dates=list(dates)

            self.x,self.y,self.ids,self.dates=time_sort(x,y,ids,dates)
            self.transform_y(transforms)

            if modelpath is None:
                now=datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d')
                modelpath='MODEL_'+now
        else: sys.exit('No input found.\n'
                        +"Specify a value for 'ifname' or specify values for x, y, ids, and date.")

        self.modelpath=modelpath

    # read input file
    def read_input(self,ifname):
        """
        read data from csv file
        
        Args:
            ifname (str): path/filename of csv file
        Returns:
            x (scipy sparse array): shape (n_compounds, nbits)
            y (np.array): shape (n_compounds, )
            ids (str list): shape (n_compounds, )
            dates (float list): assay timing as year.fraction, size (n_compounds, )
        """
        sys.stderr.write(f'Reading input data from: {ifname}\n')
        names=np.loadtxt(ifname,max_rows=1,dtype=str,delimiter=',',comments=None)
        numC=len(names)
        # ensure that at least compound ID and descriptors are specified
        if numC>=2:
            data=pd.read_csv(ifname,usecols=names)[names].to_numpy()
            ids=list(data[:,0])
            desc=data[:,1]
            # check if activity data is specified
            if numC>=3:
                y=np.array([float(d) for d in data[:,2]])
                if numC>=4:
                    dates=list(data[:,3])
                    if isinstance(dates[0],str):
                        dates= [get_date(i) for i in dates]
                else:
                    if self.dates is None: startidx=0
                    else: startidx=len(self.dates)
                    dates = [i+startidx for i in range(len(data))]
            # activity data must be present for the training set
            elif self.y is None: sys.exit('The activities for the training set are not specified in '+ifname+' \n'
                                          +'The input data file should be formatted with columns:\n'
                                          +'\t (CompoundID,SMILES,Activity,Date)')
            else:
                y=None
                dates=None
        else: sys.exit('The compound IDs and/or descriptors are not specified in '+ifname+' \n'
                       +'The input file should be formatted with columns:\n'
                       +'\t (CompoundID,Descriptors,Activity,Date)')

        # Calculate ECFP-like fingerprint
        x=np.zeros((len(desc),self.nbits))
        for i,smi in enumerate(desc):
            x[i]=get_ECFP(smi,nbits=self.nbits,radius=self.radius)

        x=sparse.csr_matrix(x,dtype=np.float32)
    
        if dates is not None:
            x,y,ids,dates=time_sort(x,y,ids,dates)
        return x,y,ids,dates

    
    # transform response
    def transform_y(self,transforms):
        """
        transform target activities, with log and/or scaling,
        return scaling parameters for future use
        
        Args:
            transforms (str): 'log', 'scale', or 'log,scale'
        Returns:
            <nothing>
        """
        ytmp=self.y
        transforms=[x.strip().lower() for x in transforms.split(',')]
        if 'log' in transforms:
            ytmp=np.log(ytmp)
        if 'scale' in transforms:
            mu=np.nanmean(ytmp)
            sigma=np.nanstd(ytmp)
            ytmp=(ytmp-mu)/sigma
        else:
            mu=0
            sigma=1
        self.y=ytmp
        self.trf={'mu':mu,'sigma':sigma,'log':'log' in transforms}
    
    def apply_transform_y(self,y):
        """
        apply stored transformation to target activities
        
        Args:
            y (np.array): target activities
        Returns:
            y (np.array): transformed activities
        """
        if self.trf['log']:
            y=np.log(y)
        y=(y-self.trf['mu'])/self.trf['sigma']
        return y

    # save model
    def save_mod(self,modelpath=None):
        """
        save/serialize model for later use
        Args:
            modelpath (str): path to save model in
        Returns:
            <nothing>
        """
        if self.x is None: sys.exit('There is no model to save. Specify inputs when initializing this object.')
        elif self.k is None: sys.exit('The model must be trained before it is saved.\n'
                                      +"See 'train_mod' function.")
        if modelpath is None: modelpath=self.modelpath
        if not os.path.exists(modelpath):
            os.mkdir(modelpath)
        sys.stderr.write(f'Saving model in: {modelpath}\n')
        # descriptors as sparse matrix
        sparse.save_npz(f'{modelpath}/x.npz',self.x)
        # targets
        np.savez(f'{modelpath}/y.npz',self.y)
        # gp hyperparameters
        with open(f'{modelpath}/gp_param.pkl','wb') as f:
            pickle.dump(self.k,f)
        # annoy index
        self.an.save(f'{modelpath}/annoy.idx')
        # ids 
        with open(f'{modelpath}/ids.txt','w') as f:
            f.write('\n'.join(self.ids))
        # training set ids
        with open(f'{modelpath}/trainids.txt','w') as f:
            f.write('\n'.join(self.trainids))
        # timing
        with open(f'{modelpath}/dates.pkl','wb') as f:
            pickle.dump(self.dates,f)
        # model parameters
        model_params={'trf':self.trf,'kneighbors':self.kneighbors,'antrees':self.antrees,
                      'chunk':self.chunk,'nbits':self.nbits,'radius':self.radius}
        with open(f'{modelpath}/model_parameters.pkl','wb') as f:
            pickle.dump(model_params,f)


    # load model
    def load_mod(self,ifname):
        """
        load saved/serialized model for use in prediction or for updating
        
        Args:
            ifname (str): path of directory containing saved model
        Returns:
            <nothing>
        """
        sys.stderr.write(f'Reading model from: {ifname}\n')
        # descriptors.  data goes in dense and comes out sparse
        self.x=sparse.load_npz(f'{ifname}/x.npz')
        # targets
        self.y=np.load(f'{ifname}/y.npz')['arr_0']
        # gp hyperparams
        with open(f'{ifname}/gp_param.pkl','rb') as f:
            self.k=pickle.load(f)
        # annoy index
        an=annoy.AnnoyIndex(self.x.shape[1],'angular')
        an.load(f'{ifname}/annoy.idx')
        self.an=an
        # ids
        with open(f'{ifname}/ids.txt','r') as f:
            self.ids=[x.strip() for x in f.readlines()]
        # trainids
        with open(f'{ifname}/trainids.txt','r') as f:
            self.trainids=[x.strip() for x in f.readlines()]
        # dates
        with open(f'{ifname}/dates.pkl','rb') as f:
            self.dates=pickle.load(f)
        # model parameters
        with open(f'{ifname}/model_parameters.pkl','rb') as f:
            model_params=pickle.load(f)
        self.trf=model_params['trf']
        self.kneighbors=model_params['kneighbors']
        self.antrees=model_params['antrees']
        self.chunk=model_params['chunk']
        self.nbits=model_params['nbits']
        self.radius=model_params['radius']


    # train kernel params & annoy index
    def train_mod(self,nfit,valmethod=None,valfrac=None,randomsplit=False):
        """
        Train kernel and annoy index
        
        Args:
            nfit (int): number of samples of x,y (taken from head of arrays) to 
                use for kernel parameter training.  
            valmethod (str): method to validate model. If None, model is not validated
            valfrac (float): fraction of samples of x,y (taken from tail of arrays)
                to use for validating the model
            randomsplit (bool): if True, the samples are randomly shuffled before
                training and validating
        Retuns:
            if validating:
                pred (pd.DataFrame): columns given as 'CompoundID' for identifiers,
                    'obs' for observed activities if provided, 'pred' for the predictions,
                    'pred_sd' for the uncertainty on those predictions
                r2 (float): squared pearson correlation of predicted vs observed
            otherwise:
                <nothing>
        """
        if self.x is None and self.y is None:
            sys.exit('There is no model to train.\n'
                     +'Ensure initial descriptor and activity data have been specified.')
        nsamp=self.x.shape[0]

        # if not validating
        if valmethod is None:
            nfit=min(nfit,nsamp)
            if nfit>self.chunk: nfit=self.chunk*(nfit//self.chunk)
            self.trainids=self.ids[:nfit]
            self.an=self.get_annoy(self.x)
            self.train_kernel(nfit)
        # validation
        elif 'contemp' in valmethod or 'chunk' in valmethod:
            nval=int(nsamp*valfrac)
            ntrain=nsamp-nval
            nfit=min(nfit,ntrain)
            self.trainids=self.ids[:nfit]
            if nfit>self.chunk: nfit=self.chunk*(nfit//self.chunk)
            # for random rather than time-split
            if randomsplit:
                yorig=self.y
                xorig=self.x
                idshuf=list(range(0,nsamp))
                random.shuffle(idshuf)
                self.x=self.x[idshuf]
                self.y=self.y[idshuf]

            # train GP
            self.train_kernel(nfit)
            # build annoy index
            if 'contemp' in valmethod:
                self.an=self.get_annoy(self.x)
            elif 'chunk' in valmethod:
                self.an=self.get_annoy(self.x[:ntrain])
            pred,r2=self.validate_mod(nval,valmethod)

            # reset annoy index and x,y arrays
            if 'chunk' in valmethod or randomsplit:
                self.an=self.get_annoy(self.x)
            if randomsplit:
                self.y=yorig
                self.x=xorig

            return pred,r2
        else: sys.exit(str(valmethod)+" is not a valid value for 'valmethod'\n"
                       +'Options are:\n'
                       +"\t 'contemp', 'contemp week', 'chunk', and None")

    def train_kernel(self,nfit):
        """
        train new model's kernel
        Args:
            nfit (int): number of samples of x,y (taken from head of arrays) to 
                use for kernel parameter training.  
        Retuns:
            <nothing>
        """
        kernel= ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-03, 1000.0))*\
                        RBF(length_scale=.5, length_scale_bounds=(.01, 2)) + WhiteKernel(noise_level=.2, noise_level_bounds=(.01, 2))
        gp=list()
        chunk=min(nfit,self.chunk)
    
        xtrain=self.x[:nfit].toarray()
        ytrain=self.y[:nfit]
        # split up sample into subsamples of size 'chunk'
        for i in range(nfit//chunk):
            tmp=GaussianProcessRegressor(kernel=kernel)
            tmp.fit(xtrain[(i*chunk):((i+1)*chunk),:],ytrain[(i*chunk):((i+1)*chunk)])
            gp.append(tmp)
        ks=pd.DataFrame([x.kernel_.get_params() for x in gp])[['k1__k1__constant_value','k1__k2__length_scale','k2__noise_level']]
        if len(gp)>1:
            # get parameter centroid
            m=ks.median()
            mn=m/np.sqrt((m**2).sum())
            ksn=ks.to_numpy()/np.sqrt((ks**2).sum(axis=1)).values.reshape((-1,1))
            ctr=mn.dot(ksn.T).argmax()
            par=gp[ctr].kernel_.get_params()
            m=[par['k1__k1__constant_value'],par['k1__k2__length_scale'],par['k2__noise_level']]
            # freeze kernel params so not changed during prediction 'fit'
            k= ConstantKernel(constant_value=m[0], constant_value_bounds=(m[0], m[0]))*\
                RBF(length_scale=m[1], length_scale_bounds=(m[1],m[1])) + WhiteKernel(noise_level=m[2], noise_level_bounds=(m[2],m[2]))
        else:
            m=[ks['k1__k1__constant_value'][0],ks['k1__k2__length_scale'][0],ks['k2__noise_level'][0]]
            k= ConstantKernel(constant_value=m[0], constant_value_bounds=(m[0], m[0]))*\
                RBF(length_scale=m[1], length_scale_bounds=(m[1],m[1])) + WhiteKernel(noise_level=m[2], noise_level_bounds=(m[2],m[2])) 
        self.k=k


    # annoy index 
    def get_annoy(self,xan):
        """
        build annoy index
        
        Args:
            xan (scipy sparse array): normalized fingerprints
        Returns:
            an (annoy index): index for identifying approximate neighbors among training set
        """
        an=annoy.AnnoyIndex(xan.shape[1],'angular')
        for i in range(xan.shape[0]):
            an.add_item(i,xan[i,:].toarray().reshape((-1,)))
        an.build(self.antrees)
        return an

    def predict_contemp(self,testin,offset):
        """
        predict using contemporaneous time split: each sample predicted based on
            all preceding samples
        
        Args:
            testin (int list): list of indices in x to predict
            offset (float): fractional time to separate prior molecules and molecule to be predicted
        Returns:
            pred (np.arrray): predicted values of x with indices testin
            predstd (np.array): corresponding standard deviations of predictions
        """
        nsamp=self.x.shape[0]
        ntest=len(testin)
        gpknn=GaussianProcessRegressor(kernel=self.k)
        pred=np.zeros(ntest)
        predstd=np.zeros(ntest)
        for j,i in enumerate(testin):
            neighbors=5*self.kneighbors # some neighbors may be from future, so initially accumulate more than KNEIGHBORS
            neigh=[x for x in self.an.get_nns_by_item(i,min(nsamp,neighbors)) if self.dates[x] < self.dates[i]-offset][:self.kneighbors]
            while len(neigh)<self.kneighbors and neighbors<nsamp:  # if first neighbors all later than query, increase search scope
                neighbors*=2
                neigh=[x for x in self.an.get_nns_by_item(i,min(nsamp,neighbors)) if self.dates[x] < self.dates[i]-offset][:self.kneighbors]
            if len(neigh)<self.kneighbors: # still? somebody gave us test data predating any training; make due
                neigh=[x for x in self.an.get_nns_by_item(i,self.kneighbors)]
            warnings.simplefilter("ignore") # sklearn gp rather whiny

            xtrain=self.x[neigh,:].toarray()
            xtest=self.x[i,:].toarray()
            
            gpknn.fit(xtrain,self.y[neigh])   # actually no 'fitting' going on, just cholesky decomposition
            pred[j],predstd[j]=gpknn.predict(xtest,return_std=True)
            warnings.simplefilter("default")
        return pred, predstd

    def predict_chunk(self,xtest):
        """
        predict activity of new compounds given model (kernel and neighbor index)
        
        Args:
            xtest (scipy sparse array): normalized fingerprints to make predictions on
        Returns:
            pred (np.array): transformed predictions for x
            predstd (np.array): transformed uncertainty predictions for x
        """
        nsamp=xtest.shape[0]
        gpknn=GaussianProcessRegressor(kernel=self.k)
        pred=np.zeros(nsamp)
        predstd=np.zeros(nsamp)
        for i in range(nsamp):
            xtmp=xtest[i,:].toarray().reshape(-1,)
            neigh=self.an.get_nns_by_vector(xtmp,self.kneighbors) 
            warnings.simplefilter("ignore") # sklearn gp rather whiny
            gpknn.fit(self.x[neigh,:].toarray(),self.y[neigh])  # 'model' x in sparse format
            pred[i],predstd[i]=gpknn.predict([xtmp],return_std=True) # pred, sd
            warnings.simplefilter("default")
        return pred, predstd

    # validate model
    def validate_mod(self,validate,valmethod):
        """
        validate model using contemporaneous time split: each sample predicted based on
            all preceding samples
        
        Args:
            validate (int or int list): number of later samples in x & y to use for validation.
                or indicies of x,y samples to validate 
            valmethod (str): method to validate model
        Returns:
            pred (pd.DataFrame): columns given as 'CompoundID' for identifiers,
                'obs' for observed activities if provided, 'pred' for the predictions,
                'pred_sd' for the uncertainty on those predictions
            r2 (float): squared pearson correlation of predicted vs observed
        """
        nsamp=self.x.shape[0]
        if type(validate)==int:
            nval=min(validate,nsamp)
            start=nsamp-nval
            valix=range(start,nsamp)
        else: # list of indices
            valix=list(validate)
            nval=len(validate)
    
        # validation method
        if 'contemp' in valmethod:
            if 'week' in valmethod:
                pred,predstd=self.predict_contemp(valix,0.0188)
            else:
                pred,predstd=self.predict_contemp(valix,0)
        elif valmethod=='chunk':
            pred,predstd=self.predict_chunk(self.x[valix])
    
        if nval>2 :
            r2=linregress(pred,self.y[valix])[2]**2
        else:
            r2=None

        # prepare DataFrame of results
        pred=np.column_stack([self.y[valix],pred,predstd])
        pred=pred*self.trf['sigma']  # reverse scaling but not log transform
        pred[:,:2]=pred[:,:2]+self.trf['mu']
        if self.trf['log']:
            pred=np.exp(pred)
            colnames=['obs','pred','pred_sd_multiplicative']
        else: colnames=['obs','pred','pred_sd']
        rownames=[self.ids[x] for x in valix]
        pred=pd.DataFrame(pred,columns=colnames,index=rownames)
        pred.index.name='CompoundID'
        return pred,r2

    # predict unknowns
    def predict(self,ifname=None,x=None,y=None,ids=None,save=True,outfile=None):
        """
        predict activity of new compounds given model (kernel and neighbor index)
        
        Args:
            ifname (str): path/filename of csv file
            x (list, np.array, scipy sparse array): list of descriptors (n_compounds, nbits)
                or list of SMILES strings (n_compounds)
            y (np.array): activity values of new compounds
            ids (str list): size (n_compounds, )
            save (bool): if True, the predictions are written to file
            outfile (str): name and path of file to write predictions to
        Returns:
            predall (pd.DataFrame): predictions transformed to original scale for input compounds
                columns: CompoundID, obs (if provided), pred, pred_sd for the predicted uncertainty
            r2 (float): squared pearson correlation of predicted vs observed
        """
        # ensure model is initialized
        if self.x is None:
            sys.exit('There is no model to make predictions.\n'
                     +'The descriptor data have been not specified.')
        # read/process data to make predictions on
        if ifname is not None:
            x,y,ids,_=self.read_input(ifname)
            nsamp=x.shape[0]
        elif x is not None:
            if isinstance(x[0],str):
                xtmp=np.zeros((len(x),self.nbits))
                for i,smi in enumerate(x):
                    xtmp[i]=get_ECFP(smi,nbits=self.nbits,radius=self.radius)
                x=xtmp
            if x.shape[1]!=self.nbits: sys.exit("'x' is invalid.\n"
                           +"It must be a list of SMILES strings or a list of fingerprints, with fingerprint length matching 'nbits'")
            if not sparse.issparse(x): x=sparse.csr_matrix(x,dtype=np.float32)
            nsamp=x.shape[0]

            if y is not None:
                if not isinstance(y,np.ndarray): y=np.array(y)

            if ids is None: ids=[str(i) for i in range(nsamp)]
            elif not isinstance(ids,list): ids=list(ids)
        else:
            sys.exit('Input to be predicted has not been specified.\n'
                     +"Ensure that 'ifname' or 'x' have been passed to 'predict'.")
        x=norm_x(x) 
 
        pred,predstd = self.predict_chunk(x)
        predall=np.column_stack((pred*self.trf['sigma']+self.trf['mu'],predstd*self.trf['sigma']))
        if self.trf['log']:
            predall=np.exp(predall)
            colnames=['pred','pred_sd_multiplicative']
        else: colnames=['pred','pred_sd']

        r2=0 
        # if measurements are available, get r2
        if y is not None and len(y.shape)==1 and y.shape[0]==predall.shape[0]:
            predall=np.column_stack((y,predall))
            if nsamp>2:
                r2=linregress(predall[:,1],y)[2]**2
                colnames=['obs']+colnames
        # prepare DataFrame and populate
        pred=pd.DataFrame(predall,index=ids,columns=colnames)
        pred.index.name="CompoundID"
        if save:
            if outfile is None: outfile=self.modelpath+'/pred.csv'
            elif not os.path.isabs(outfile): outfile=self.modelpath+'/'+outfile
            sys.stderr.write(f'Writing results to: {outfile}\n')
            with open(outfile,'w') as outf:
                pred.to_csv(outf)

        return pred,r2


    def update_mod(self,ifname=None,x=None,y=None,ids=None,dates=None,valmethod=None,save=True,outfile=None):
        """
        Update existing model with new samples.  Kernel parameters left unchanged, only 
            neighbor index and compound descriptors/activity updated
            
        Args:
            ifname (str): path/filename of csv file with new data
            x (list, np.array, scipy sparse array): list of descriptors (n_compounds, nbits)
                or list of SMILES strings (n_compounds)
            y (np.array): activity values of new molecules
            ids (str list): ids of new molecules
            dates (float list): assay timing as year.fraction of new molecules
            valmethod (str): method to validate model. If None, model is not validated
            save (bool): if True and valmethod is provided, the predictions are written to file
            outfile (str): name and path of file to write predictions to
        Returns:
            if validate:
                valpred (pd.DataFrame): predictions for x.  predictions transformed to original scale.
                    columns given as 'CompoundID' for identifiers, 'obs' for observed activities if provided,
                    'pred' for the predictions, 'pred\_sd' for the uncertainty on those predictions.
                r2 (float): squared pearson correlation of predicted vs observed
                numup (int): number of times the model is updated during the validation process
                    e.g. if 'chunk week', numup is the number of week-long chunks in the new sample
            otherwise:
                <nothing>
        """
        # ensure model has been initialized
        if self.x is None:
            sys.exit('There is no model to update.\n'
                     +'Ensure initial descriptor and activity data have been specified.')
        oldnsamp=len(self.ids)
        # read/process data to update model with
        if ifname is not None:
            x,y,ids,dates=self.read_input(ifname)
            nsamp=x.shape[0]
        elif x is not None and y is not None:
            if isinstance(x[0],str):
                xtmp=np.zeros((len(x),self.nbits))
                for i,smi in enumerate(x):
                    xtmp[i]=get_ECFP(smi,nbits=self.nbits,radius=self.radius)
                x=xtmp
            if x.shape[1]!=self.nbits: sys.exit("'x' is invalid.\n"
                           +"It must be a list of SMILES strings or a list of fingerprints, with fingerprint length matching 'nbits'")
            if not sparse.issparse(x): x=sparse.csr_matrix(x,dtype=np.float32)
            nsamp=x.shape[0]

            if not isinstance(y,np.ndarray): y=np.array(y)

            if ids is None:
                ids=[str(oldnsamp+i) for i in range(nsamp)]
            elif not isinstance(ids,list): ids=list(ids)

            if dates is None:
                dates=[self.dates[-1]+i for i in range(nsamp)]
            elif not isinstance(dates,list): dates=list(dates)
            x,y,ids,dates=time_sort(x,y,ids,dates)
        else:
            sys.exit('Input to update the model with has not been specified.\n'
                     +"Ensure that 'ifname' or (x,y,ids,dates) have been passed to 'update_mod'.")

        sys.stderr.write(f'{nsamp} new samples\n')

        x=norm_x(x) 
        y=self.apply_transform_y(y)

        iddict={j:i for i,j in enumerate(ids)}

        # delete old copies of repeated ids
        keepidx=[i for i,j in enumerate(self.ids) if j not in iddict]
        oldx=self.x[keepidx]
        oldy=self.y[keepidx]
        olddates=[self.dates[i] for i in keepidx]
        oldids=[self.ids[i] for i in keepidx]
    
        # update fingerprint, y, dates, ids
        self.x=sparse.vstack([oldx,x])
        self.y=np.concatenate([oldy,y])
        self.dates=olddates+dates
        self.ids=oldids+ids

        # validation procedure
        if valmethod is None:
            self.an=self.get_annoy(self.x)
        else:
            valpred,r2,numup=self.validate_update_mod(nsamp,valmethod)
            # save the validation of the updated model
            if save:
                if outfile is None:
                    now=datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d')
                    outfile=self.modelpath+'/validation_'+now+'.csv'
                elif not os.path.isabs(outfile): outfile=self.modelpath+'/'+outfile
                sys.stderr.write(f'Writing results to: {outfile}\n')
                with open(outfile,'w') as outf:
                    valpred.to_csv(outf)

            return valpred,r2,numup


    def validate_update_mod(self,newnsamp,valmethod):
        """
        Validate the updated model according to valmethod. 
            Kernel parameters left unchanged, only update neighbor index.
            Retrieve the prediction data and corresponding R2
            
        Args:
            newnsamp (int): number of new samples
            valmethod (str): method to validate and update model
        Returns:
            valpred (pd.DataFrame): predictions for x.  predictions transformed to original scale.
                columns given as 'CompoundID' for identifiers, 'obs' for observed activities if provided,
                'pred' for the predictions, 'pred\_sd' for the uncertainty on those predictions.
            r2 (float): squared pearson correlation of predicted vs observed
            numup (int): number of times the model is updated during the validation process
                e.g. if 'chunk week', numup is the number of week-long chunks in the new sample
        """
        nsamp=self.x.shape[0]
        oldnsamp=nsamp-newnsamp
        # contemporaneous validation
        if 'contemp' in valmethod:
            self.an=self.get_annoy(self.x)
            valix=range(oldnsamp,oldnsamp+newnsamp)
            if 'week' in valmethod: offset=0.0188
            else: offset=0
            pred,predstd=self.predict_contemp(valix,offset)
            numup=1 
        # chunk validation
        elif 'chunk' in valmethod:
            chunkind=[oldnsamp]
            if valmethod=='chunk week':
                indate=self.dates[oldnsamp]
                for i in range(newnsamp):
                    if self.dates[oldnsamp+i]-indate>=0.0188:
                        chunkind.append(oldnsamp+i)
                        indate=self.dates[oldnsamp+i]
            elif len(valmethod.split())==2 and valmethod.split()[1].isdigit():
                pchunk=int(valmethod.split()[1])
                nchunk=newnsamp//pchunk
                for i in range(1,pchunk):
                    chunkind.append(oldnsamp+i*nchunk)
            chunkind.append(oldnsamp+newnsamp)
            numup=len(chunkind)-1

            for i in range(len(chunkind)-1):
                # update annoy index, not needed for first chunk
                if i>0: self.an=self.get_annoy(self.x[:chunkind[i]])

                predi,predstdi=self.predict_chunk(self.x[chunkind[i]:chunkind[i+1]])
                if i==0:
                    pred=predi
                    predstd=predstdi
                else:
                    pred=np.concatenate((pred,predi))
                    predstd=np.concatenate((predstd,predstdi))

            self.an=self.get_annoy(self.x)
    
        ynew=self.y[oldnsamp:]
        r2=linregress(pred,ynew)[2]**2

        # form DataFrame of results
        valpred=np.column_stack((ynew,pred,predstd))
        valpred=valpred*self.trf['sigma']
        valpred[:,:2]=valpred[:,:2]+self.trf['mu']
        if self.trf['log']:
            valpred=np.exp(valpred)
            colnames=['obs','pred','pred_sd_multiplicative']
        else: colnames=['obs','pred','pred_sd']
        valpred=pd.DataFrame(valpred,index=self.ids[oldnsamp:],columns=colnames)    
        valpred.index.name="CompoundID"

        return valpred,r2,numup


    # prospective
    def prosp_pred(self,ifname=None,x=None,y=None,ids=None,save=True,outfile=None):
        """
        predictions for (possibly) new data, x.  if already seen (in ids),
            references only predating samples. if in original training set 
            (trainids), no prediction returned
            
        Args:
            ifname (str): path/filename of csv file with new data
            x (list, np.array, scipy sparse array): list of descriptors (n_compounds, nbits)
                or list of SMILES strings (n_compounds)
            y (np.array): activity values of new molecules
            ids (str list): ids of new molecules
            save (bool): if True, the predictions are written to file
            outfile (str): name and path of file to write predictions to
        Returns:
            out (pd.DataFrame): predictions for x
                columns: CompoundID, pred, pred_sd, obs, train_nopred, retro_pred
        """
        if self.x is None:
            sys.exit('There is no model to make prospective predictions from.\n'
                     +'Ensure initial descriptor and activity data have been specified.')
        firstiddict={j:i for i,j in enumerate(self.trainids)}
        previddict={j:i for i,j in enumerate(self.ids)}
        oldnsamp=len(self.ids)

        # read/process data to make prospective predictions on
        if ifname is not None:
            x,y,ids,dates=self.read_input(ifname)
            nsamp=x.shape[0]
        elif x is not None:
            if isinstance(x[0],str):
                xtmp=np.zeros((len(x),self.nbits))
                for i,smi in enumerate(x):
                    xtmp[i]=get_ECFP(smi,nbits=self.nbits,radius=self.radius)
                x=xtmp
            if x.shape[1]!=self.nbits: sys.exit("'x' is invalid.\n"
                           +"It must be a list of SMILES strings or a list of fingerprints, with fingerprint length matching 'nbits'")
            if not sparse.issparse(x): x=sparse.csr_matrix(x,dtype=np.float32)
            nsamp=x.shape[0]

            if y is not None:
                if not isinstance(y,np.ndarray): y=np.array(y)

            if ids is None:
                ids=[str(oldnsamp+i) for i in range(nsamp)]
            elif not isinstance(ids,list): ids=list(ids)
        else:
            sys.exit('Input to make predictions on has not been specified.\n'
                     +"Ensure that 'ifname' or (x,ids) have been passed to 'prosp_pred'.")
        x=norm_x(x)

        # index of ids that are in first training set
        nopredidx=[j for j,i in enumerate(ids) if i in firstiddict]
        # index in ids and in previddict of those not in firstiddict, but in previddict
        validx=[[j,previddict[i]] for j,i in enumerate(ids) if i in previddict and i not in firstiddict]
        # index of new ids to make predictions for
        predidx=[j for j,i in enumerate(ids) if i not in previddict]

        if y is None:
            y=np.full(nsamp,None)
            for j,i in enumerate(ids):
                if i in previddict:
                    ytmp=self.y[previddict[i]]*self.trf['sigma']+self.trf['mu']
                    if self.trf['log']: ytmp=np.exp(ytmp)
                    y[j]=ytmp

        # initialize output
        out=np.full((nsamp,5),None) # cols: pred,1sd,obs,firsttrain,prosp

        # observed
        if len(y)==nsamp:
            out[validx+predidx+nopredidx,2]=y[validx+predidx+nopredidx]
        
        # get predictions from those already tested
        if len(validx)>0:
            validx,valprevidx=zip(*validx)
            validx,valprevidx=list(validx),list(valprevidx)
            valpred,_=validate_mod(valprevidx,'contemp') # valpred cols: obs, pred, sigma
            valpred=valpred.as_matrix()
            out[validx,4]=1 # indicator column: prospective pred
            out[validx,:2]=valpred
        # get predictions for new compounds
        if len(predidx)>0:
            pred,predstd=self.predict_chunk(x[predidx,:])
            pred=np.column_stack((pred,predstd))
            pred=pred*self.trf['sigma']
            pred[:,0]=pred[:,0]+self.trf['mu']
            if self.trf['log']: # unlog
                pred=np.exp(pred)
            out[predidx,:2]=pred
        # for compounds in first training set
        if len(nopredidx)>0:
            out[nopredidx,3]=1 # indicator column: first train set, no pred

        # prepare output DataFrame
        if self.trf['log']:
            colnames=['pred','pred_sd_multiplicative','obs','train_nopred','retro_pred']
        else: colnames=['pred','pred_sd','obs','train_nopred','retro_pred']
        out=pd.DataFrame(out,index=ids,columns=colnames)    
        out.index.name="CompoundID"
        if save:
            if outfile is None:
                now=datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d')
                outfile=self.modelpath+'/prospective_'+now+'.csv'
            elif not os.path.isabs(outfile): outfile=self.modelpath+'/'+outfile
            sys.stderr.write(f'Writing results to: {outfile}\n')
            with open(outfile,'w') as outf:
                out.to_csv(outf)

        return out


# remove invalid activity values
def valid_values(x,y,ids,dates):
    """
    removes x,y,ids, and dates entries associated with y values that are nan
    
    Args:
        x (np.array or scipy.sparse): shape (n_compounds, nbits)
        y (np.array): shape (n_compounds, )
        ids (str list): shape (n_compounds, )
        dates (float list): assay timing as year.fraction, shape (n_compounds, )
    Returns:
        x (np.array or scipy.sparse): shape (n_compounds, nbits)
        y (np.array): shape (n_compounds, )
        ids (str list): shape (n_compounds, )
        dates (float list): assay timing as year.fraction, shape (n_compounds, )
    """
    sel=np.invert(np.isnan(y))
    y=y[sel]
    x=x[sel,:]
    ids=list(np.array(ids)[sel])
    dates=list(np.array(dates)[sel])
    return x,y,ids,dates
    
# sort by ascending date
def time_sort(x,y,ids,dates):
    """
    sort x,y,ids by increasing dates 
    
    Args:
        x (np.array or scipy.sparse): shape (n_compounds, nbits)
        y (np.array): shape (n_compounds, )
        ids (str list): shape (n_compounds, )
        dates (float list): assay timing as year.fraction, shape (n_compounds, )
    Returns:
        x (np.array or scipy.sparse): shape (n_compounds, nbits)
        y (np.array): shape (n_compounds, )
        ids (str list): shape (n_compounds, )
        dates (float list): assay timing as year.fraction, shape (n_compounds, )
    """
    sortidx,dates=zip(*sorted(enumerate(dates),key=lambda p: p[1]))
    sortidx=list(sortidx)
    x=x[sortidx]
    y=y[sortidx]
    ids=[ids[i] for i in sortidx]
    dates=list(dates)
    return x,y,ids,dates
    
# process date string
def get_date(d):
    """
    parse date in either fractional format (year.fraction), or other common format
        (year-month-day, day/month/year, etc).  Alternatively dummy integer indices
        passed through.
        
    Args:
        date (str): date string
    Returns:
        fractional date (float): sortable respresentation 
    """
    if re.match(r'(19|20)\d\d\.\d{3,4}',d):  # year.fraction format
        return float(d)
    if re.match(r'^\d+$',d):  # integer index, leave as is
        return float(d)
    try:
        return fract_date(d)
    except:  
        return fract_date('Jan 1, 1970')
        
# generic date format to fractional date
def fract_date(datestring):
    dd=pd.to_datetime(datestring,errors='coerce')
    y,m,d,md=dd.year,dd.month,dd.day,dd.days_in_month
    mf=float(m-1)/12  #0-0.916
    df=float(d-1)/md  #0-0.968
    out=np.round(y+mf+df/12,4)
    if np.isnan(out):
        raise ValueError()
    return out
    
# normalize input vectors
def norm_x(x):
    """
    normalize x such that each row has length 1
        
    Args:
        x (scipy sparse array): input fingerprint matrix
    Returns:
        x (scipy sparse array): fingerprint matrix normalized to L2 norm = 1
    """
    newxs=x/np.sqrt(x.power(2).sum(axis=1))
    return sparse.csr_matrix(newxs,dtype=np.float32)

# descriptor generation
def get_ECFP(smi,nbits=1024,radius=3):
    """
    calculates ECFP-like descriptors from input moleule

    Args:
        smi (str): molecule represented as SMILES string
        nbits (int): number of bits to hash/fold fingerprint to
        radius (int): distance in bonds to expand from each atom in creating FP

    Returns:
        out (np.uint8 array): length nbits array of 0/1 bits representing folded FP
    """

    mol=MolFromSmiles(smi,sanitize=False)
    SanitizeMol(mol,SanitizeFlags.SANITIZE_SYMMRINGS|SanitizeFlags.SANITIZE_SETAROMATICITY,catchErrors=True)
    fp=GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

    return np.array([x for x in fp],dtype=np.uint8)


