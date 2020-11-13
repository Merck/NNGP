# Nearest Neighbor Gaussian Process
This repository contains the python implementation of the Nearest Neighbor Gaussian Process method described in this paper:

DiFranzo, A.; Sheridan, R. P.; Liaw, A.; Tudor, M. Nearest Neighbor Gaussian Process for Quantitative Structure-Activity Relationships. *J. Chem. Inf. Model.* **2020**, *60*(10), 4653-4663. [DOI: 10.1021/acs.jcim.0c00678](https://doi.org/10.1021/acs.jcim.0c00678)

## Table of Contents
- [Requirements](#requirements)
- [Basic Usage](#basic-usage)
  * [Initialize a model](#initialize-a-model)
  * [Train a model](#train-a-model)
  * [Make predictions](#make-predictions)
  * [Update a model](#update-a-model)
- [NNGP API](#nngp-api)
  * [NNGaussianProcess class structure](#nngaussianprocess-class-structure)
  * [NNGaussianProcess class structure with RDKit](#nngaussianprocess-class-structure-with-rdkit)
  * [Primary NNGaussianProcess methods](#primary-nngaussianprocess-methods)


## Requirements
* annoy >= 1.16.3
* numpy >= 1.18.4
* pandas >= 1.1.1
* scikit-learn >= 0.22.2.post1
* scipy >= 1.4.1

If using the 'nngp\_rdkit.py' version of the code, this will also be required:
* rdkit >= 2020.03.5


## Basic Usage
This section details the basic usage and functionality of NNGP. A more comprehensive explanation of the API is available in the next section. See 'example\_nngp.py' for explicit examples and variations. The sample data for that example is included, 'METAB\_training.csv' and 'METAB\_test.csv'. The data has been split so the training set has the earliest 75% of tested compounds and the test set has the more recent 25%. These files include the following columns:
* CompoundID: unique compound identifiers
* DESCRIPTORS: a list of AP,DP descriptors for each compound, formatted as '<Descriptor name 1>:<value 1> <Descriptor name 2>:<value 2> ...'
* ACTIVITY: metabolism measurement, percent remaining after 30 minutes of microsomal incubation
* DATE: the compound testing date, formatted as fractional-years

These data sets are publicly available. The 'DESCRIPTORS' and 'ACTIVITY' columns are available from the Supporting Information section of [Extreme Gradient Boosting as a Method for Quantitative Structure-Activity Relationships](https://doi.org/10.1021/acs.jcim.0c00029), listed as 'ci0c00029\_si\_019.zip'. The 'DATE' column is available from the Supporting Information section of [Nearest Neighbor Gaussian Process for Quantitative Structure-Activity Relationships](https://doi.org/10.1021/acs.jcim.0c00678), listed as 'METAB\_dates.txt' inside 'ci0c00678\_si\_001.zip'. Both data sets include the unique compound identifiers which can be used to merge the descriptor, activity, and date information together.


### Initialize a model
There are two versions of the code. The first, 'nngp.py', is intended for descriptors which are variable in length, such as AP,DP descriptors. This code expects input csv files to be formatted with columns 'CompoundID,Descriptors,Activity,Date', where the 'Descriptors' is a list of descriptor names followed by its value separated by ':', e.g. 'Desc\_1:1 Desc\_2:3 Desc\_3:1 ...etc'. The second version, 'nngp\_rdkit.py', uses RDKit to calculate folded ECFPs from SMILES strings. This code expects the input csv files to be formatted with columns 'CompoundsID,SMILES,Activity,Date'. However, in both cases, data can be entered as arguments rather than read from an input file to allow for other descriptor types. 

There are three ways to initialize a model:
1. From a data file:
  * `mymod=nngp.NNGaussianProcess(ifname=<file to be read>, modelpath=<dir to save to>)`
2. From arguments:
  * `mymod=nngp.NNGaussianProcess(x=<list of descriptors>, y=<list of activities>, ids=<list of compound ids>, dates=<list of dates>, modelpath=<dir to save to>)`
3. From a trained model:
  * `mymod=nngp.NNGaussianProcess(ifname=<dir containing trained model>)`


### Train a model
If initializing from a data file or arguments, the initialization step only imports the data and sets hyperparameters. No training is done.

The model can be trained on part of the data and validated. For example, the following will train on the first 75% and validate on the last 25% of the input data, using the first 500 observations to fit model parameters:
```python
mymod.train_mod(500, valmethod='chunk', valfrac=0.25)
```
Otherwise, the model can be trained on the entire sample:
```python
mymod.train_mod(500)
```
With a trained model in hand, it can now be saved:
```python
mymod.save_mod()
```


### Make predictions
Given new compounds, we can make predictions from an input file (formatted in the same way as the training file):
```python
mymod.predict(ifname=<file with new sample>, save=True, outfile=<file to save predictions to>)
```
Or, descriptors can be passed directly as an argument, without any additional data:
```python
mymod.predict(x=<list of descriptors>, save=True, outfile=<file to save predictions to>)
```
If the new sample possibly has a mix of new compounds and those already seen, the following command can be used. If a compound isn't part of the model, a prediction will be made. If it has been seen but is not part of the initial sample used to fit the kernel parameters, only compounds which predate it are used to make the prediction. If the compound was part of that initial training set, no prediction is made. The distinction between the first two options is tagged in the output.
```python
mymod.prosp_pred(ifname=ftest, save=True, outfile=<csv file to write predictions>)
```
In all cases, the predictions are returned by these functions and written to a file, if requested. If observed data is provided, an R2 value is also returned.


### Update a model
The model can also be updated by providing new data:
```python
mymod.update_mod(ifname=ftest)
```
The model can also be validated as it updates. For example, update and test in 4 chunks:
```python
mymod.update_mod(ifname=<file with new sample>, valmethod='chunk 4')
```
Or update and test predictions after each week's worth of data:
```python
mymod.update_mod(ifname=<file with new sample>, valmethod='chunk week')
```
Or contemporaneously:
```python
mymod.update_mod(ifname=<file with new sample>, valmethod='contemp week')
```
Finally, we can save this updated model to some (possibly new) directory:
```python
mymod.save_mod(modelpath=<dir to save model in>)
```




## NNGP API

### NNGaussianProcess class structure
*class* **nngp.NNGaussianProcess**
```python
nngp.NNGaussianProcess(ifname=None, x=None, y=None, ids=None, dates=None, deslist=None, kneighbors=100, antrees=50, chunk=500, transforms='scale', dynamic_descriptor=True, modelpath=None)
```
Initializes model by reading data inputs or load previously trained model. **ifname** or (**x,y,ids,dates**) must be provided, not both.

Parameters:
* **ifname** (str): path of input csv data file (with columns 'CompoundID,Descriptors,Activity,Date') or directory containing trained model
* **x** (float list: n\_samples, n\_descriptors): list of descriptors for each compound
* **y** (float list: n\_samples): activity for each compound
* **ids** (str list: n\_samples): compound identifiers
* **dates** (str or float list: n\_samples): testing date for each compound, either as fractional year or in DD-MMM-YYYY
* **deslist** (str list: n\_descriptors): list of unique descriptor names to use. If none are specified, the unique set from the input is used
* **kneighbors** (int): the number of nearest neighbors to use for making predictions
* **antrees** (int): the number of trees for annoy to build
* **chunk** (int): the number of compounds to train the GP on at one time
* **transforms** (str): how to transform activity data
  * 'scale': rescale to have zero mean and unit standard deviation
  * 'log': log transform
  * 'log,scale': log transform then rescale
* **dynamic\_descriptors** (bool): if True, the descriptor list is updated as new compounds are incorporated. Otherwise, the descriptors used are fixed.
* **modelpath** (str): directory to save model files in



### NNGaussianProcess class structure with RDKit
*class* **nngp\_rdkit.NNGaussianProcess**
```python
nngp_rdkit.NNGaussianProcess(ifname=None, x=None, y=None, ids=None, dates=None, kneighbors=100, antrees=50, chunk=500, nbits=1024, radius=3, transforms='scale', modelpath=None)
```
Initializes model by reading data inputs or load previously trained model. **ifname** or (**x,y,ids,dates**) must be provided, not both.

Parameters:
* **ifname** (str): path of input csv data file (with columns 'CompoundID,SMILES,Activity,Date') or directory containing trained model
* **x** (str list: n\_samples): list of SMILES strings
* **y** (float list: n\_samples): activity for each compound
* **ids** (str list: n\_samples): compound identifiers
* **dates** (str or float list: n\_samples): testing date for each compound, either as fractional year or in DD-MMM-YYYY
* **kneighbors** (int): the number of nearest neighbors to use for making predictions
* **antrees** (int): the number of trees for annoy to build
* **chunk** (int): the number of compounds to train the GP on at one time
* **nbits** (int): the number of bits to fold the ECFP to
* **radius** (int): the radius of the ECFPs (e.g. 3 corresponds to ECFP6)
* **transforms** (str): how to transform activity data
  * 'scale': rescale to have zero mean and unit standard deviation
  * 'log': log transform
  * 'log,scale': log transform then rescale
* **modelpath** (str): directory to save model files in



### Primary NNGaussianProcess methods
The following methods for both versions of the code have the same names and arguments. However, when using the nngp\_rdkit version, **x** can be given as a list of SMILES strings. The methods are listed in alphabetical order. See the source code for additional methods and details.

#### load\_mod
```python
load_mod(ifname)
```
Load a saved model. Useful to reinitialize a model to its original saved state.

Parameters:
* **ifname** (str): path of directory containing saved model


#### predict
```python
predict(ifname=None, x=None, y=None, ids=None, save=True, outfile=None)
```
Make predictions for inputs in **ifname** or **x**. No updates are made to the model, thus predictions are based solely on the training data.

Parameters:
* **ifname** (str): path of directory containing saved model
* **x** (float list: n\_samples, n\_descriptors): list of descriptors for each compound
* **y** (float list: n\_samples): activity for each compound
* **ids** (str list: n\_samples): compound identifiers
* **dates** (str or float list: n\_samples): testing date for each compound, either as fractional year or in DD-MMM-YYYY
* **save** (bool): if True, the predictions are saved to a csv file
* **outfile** (str): path of csv file to write predictions to

Returns:
* **preds** (pd.DataFrame): predictions transformed to original scale for inputs, with columns given as 'CompoundID' for identifiers, 'obs' for observed activities if provided, 'pred' for the predictions, 'pred\_sd' for the uncertainty on those predictions.
* **r2** (float): squared pearson correlation of predicted vs observed


#### prosp\_pred
```python
prosp_pred(ifname=None, x=None, y=None, ids=None, save=True, outfile=None)
```
Make predictions for (possibly) new inputs in **ifname** or **x**. No updates are made to the model, thus predictions are based solely on the training data. If a compound is part not part of the model, a prediction is made. If a compound is part of the model but not part of the initial sample used to fit the kernel parameters, only compounds which predate it are used to make the prediction. If the compound is part of the initial sample used to fit the kernel parameters, no prediction is made.

Parameters:
* **ifname** (str): path of directory containing saved model
* **x** (float list: n\_samples, n\_descriptors): list of descriptors for each compound
* **y** (float list: n\_samples): activity for each compound
* **ids** (str list: n\_samples): compound identifiers
* **save** (bool): if True, the predictions are saved to a csv file
* **outfile** (str): path of csv file to write predictions to

Returns:
* **preds** (pd.DataFrame): predictions transformed to original scale for inputs, with columns given as 'CompoundID' for identifiers, 'pred' for the predictions, 'pred\_sd' for the uncertainty on those predictions, 'obs' for observed activities if provided, 'train\_nopred' where a value of '1' means no prediction is made, 'retro\_pred' where a value of '1' means the compound is already part of the model and only predating compounds were used.


#### save\_mod
```python
save_mod(modelpath=None)
```
Save the current state of the model.

Parameters:
* **modelpath** (str): new path to save model in, if none is provided it is saved into 'modelpath' specified during initialization


#### train\_mod
```python
train_mod(nfit, valmethod=None, valfrac=None, randomsplit=False)
```
Train the GP kernel and build annoy index. Optionally, validate the model.

Parameters:
* **nfit** (int): number of initial samples to use for kernel parameter training
* **valmethod** (str): method to validate model. If none is given, the model is not validated.
  * 'contemp': all predating compounds (including those in validation set) are accessible to make a prediction
  * 'contemp week': only compounds which predate the query by at least a week are accessible
  * 'chunk': compounds in the validation set are not used to make predictions
* **valfrac** (float): fraction of samples (taken from tail of arrays) to use for validation
* **randomsplit** (bool): if True, the samples are randomly shuffled before training and validating

Returns:
* if valmethod is provided:
  * **preds** (pd.DataFrame): predictions transformed to original scale for inputs, with columns given as 'CompoundID' for identifiers, 'obs' for observed activities if provided, 'pred' for the predictions, 'pred\_sd' for the uncertainty on those predictions.
  * **r2** (float): squared pearson correlation of predicted vs observed
* otherwise: nothing


#### update\_mod
```python
update_mod(ifname=None, x=None, y=None, ids=None, dates=None, valmethod=None, save=True, outfile=None)
```
Update existing model with new samples. Kernel parameters are left unchanged, only neighbor index and compound descriptors/activities are updated.

Parameters:
* **ifname** (str): path of input csv data file (with columns 'CompoundID,Descriptors,Activity,Date')
* **x** (float list: n\_samples, n\_descriptors): list of descriptors for each compound
* **y** (float list: n\_samples): activity for each compound
* **ids** (str list: n\_samples): compound identifiers
* **dates** (str or float list: n\_samples): testing date for each compound, either as fractional year or in DD-MMM-YYYY
* **valmethod** (str): method to validate model. If none is given, the model is not validated. Options:
  * 'contemp': all predating compounds are accessible to make a prediction
  * 'contemp week': only compounds which predate the query by at least a week are accessible
  * 'chunk week': update model after each week's worth of data and make predictions for following week
  * 'chunk N': update model in N (roughly) equal chunks and make predictions for following chunk
* **save** (bool): if True and valmethod is provided, the predictions are saved to a csv file
* **outfile** (str): path of csv file to write predictions to

Returns:
* if valmethod is provided:
  * **preds** (pd.DataFrame): predictions transformed to original scale for inputs, with columns given as 'CompoundID' for identifiers, 'obs' for observed activities if provided, 'pred' for the predictions, 'pred\_sd' for the uncertainty on those predictions.
  * **r2** (float): squared pearson correlation of predicted vs observed
  * **numup** (int): number of times the model was updated during the validation process
* otherwise: nothing

