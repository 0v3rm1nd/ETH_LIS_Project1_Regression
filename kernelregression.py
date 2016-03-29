import pandas as pd
import numpy as np
import sklearn.linear_model as sklin
import sklearn.cross_validation as skcv
import sklearn.svm as svm
import sklearn.metrics as skmet
import sklearn.grid_search as skgs
import datetime

## You have to install the pandas modul

## Unfortunately there isn't yet a cross-validation implemented...no time :-)

## Overview
## fct: ret_time_values
##      extracts vars from given timestamp
##      define manually new features
## fct: create_design_matrix
##      puts the defined vars in one dataframe
##      define manually new features
## fct: logscore
##      evaluation function on validation set
## fct: kernelregression
##      test regression function, how good are we on test-evaluation-set
## fct: kernelregression1
##      regression function for prediction and storage of .csv file
##      here the storage functionality should be outsourced...no time left :-)



##############
## functions
def ret_time_values(timestamp):

    ##extract time_vars
    ret = pd.DataFrame()
    year = pd.DatetimeIndex(timestamp).year
    year = pd.get_dummies(year)
    year.columns = ['2013','2014','2015']
    month = pd.DatetimeIndex(timestamp).month
    month = pd.get_dummies(month)
    month.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    day = pd.DatetimeIndex(timestamp).weekday
    day = pd.get_dummies(day)
    day.columns = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    hour = pd.DatetimeIndex(timestamp).hour
    hour_d = pd.get_dummies(hour)
    hour_d.columns = ['0h','1h','2h','3h','4h','5h','6h','7h','8h','9h','10h','11h','12h', \
                    '13h','14h','15h','16h','17h','18h','19h','20h','21h','22h','23h']
    minute = pd.DatetimeIndex(timestamp).minute
    minute = pd.get_dummies(minute)
    minute.columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16',\
                      '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',\
                      '32','33','34','35','36','37','38','39','40','41','42','43','44','45','46',\
                      '47','48','49','50','51','52','53','54','55','56','57','58','59']
    second = pd.DatetimeIndex(timestamp).second
    day_int = pd.DatetimeIndex(timestamp).day
    #################
    ## define time vars manually using for regression
    ret[list(year.columns)] = year
    ret[list(month.columns)] = month
    ret[list(day.columns)] = day
    #ret[list(hour_d.columns)] = hour_d
    ret['hour'] = hour
    return ret

def create_design_matrix(V):
    X = V
    #get timestamp
    timestamp = pd.to_datetime(X['timestamp'],format='%Y-%m-%d %H:%M:%S')
    #get timestamp vars
    time_splitted = ret_time_values(timestamp)

    # B categorial
    B = pd.get_dummies(X['B'])
    B.columns = ['B1','B2','B3','B4']

    #remove timestamp and var B
    del X['timestamp']
    del X['B']
    
    #X['F*D'] = X['F']*X['D']
    #X['F*D'] = np.exp(X['F']*X['D'])
    #X['A*C'] = X['A'] * X['D']
    ## center matrix
    for i in list(X.columns):
        X[i] = (X[i] - np.mean(X[i])) / np.std(X[i])

    #add dummmy_var B and time_vars
    X[list(B.columns)] = B
    X[list(time_splitted.columns)] = time_splitted

    #define new features manually
    X['hour+A'] = X['hour'] + X['A']
    X['hour+E'] = X['hour'] + X['E']
    X['hour+F'] = X['hour'] + X['F']
    #X['hour^2'] = X['hour']**2
    #X['A*C'] = X['A'] * X['D']
    #X['F*D'] = X['F']*X['D']
    
    #eliminate features manually
    del X['F']
    del X['D']
    #del X['C']
    #del X['E']
    #del X['A']
    #del X['hour']
    return X


def logscore(y, y_hat):
    """
    input vars:
    y: Evaluation response
    y_hat: predicted response from regression
    """
    y_hat = pd.DataFrame(data=y_hat)
    y_hat = np.clip(y_hat,0,np.inf)
    logdif = np.log(1 + y) - np.log(1 + y_hat)
    return np.sqrt(np.mean(np.square(logdif)))


def logscore1(y, y_hat):
    print 'call'
    y_hat = np.clip(y_hat,0,np.inf)
    logdif = np.log(1 + y) - np.log(1 + y_hat)
    return np.sqrt(np.mean(np.square(logdif)))

def kernelregression(X_tr,y_tr,X_te,y_te):
    """
    X_tr: Training explanatory set
    y_tr: Training response set
    X_te: Evaluation set
    y_te: Evaluation response
    """
    print list(X_tr.columns)
    # regressor
    regressor = svm.SVR(kernel='rbf',degree=1,epsilon=0.1)
    #param_grid = {'degree':[1,2,3,4,5],'epsilon':[0.1,0.2,0.3,0.4,0.5]}
    param_grid = {'C':[1,2,3,4,5,6]}
    print param_grid
    neg_scorefun = skmet.make_scorer(lambda x,y: -logscore1(x,y))
    grid_search = skgs.GridSearchCV(regressor,param_grid,scoring=neg_scorefun,cv=3)
    grid_search.fit(X_tr,y_tr[0])
    best = grid_search.best_estimator_
    y_hat = best.predict(X_te)
    y_hat = np.exp(y_hat)
    print best
    return logscore(y_te,y_hat).values
    #print 'best score = ', -grid_search.best_score_
    
    
##    regressor = svm.SVR(kernel='rbf',degree=5,epsilon=0.2)
##    regressor.fit(X_tr,y_tr[0])
##    y_hat = regressor.predict(X_te)
##    y_hat = np.exp(y_hat)
##    return logscore(y_te,y_hat).values
    
def kernelregression1(X_tr,y_tr,X_te):
    """
    X_tr: Training set
    y_tr: Training response
    X_te: Validation set
    """
    # degree=5, epsilon = 0.2"
    regressor = svm.SVR(kernel='rbf',degree=1,epsilon=0.1,C=6)
    regressor.fit(X_tr,y_tr['y'])
    y_hat = regressor.predict(X_te)
    y_hat = np.exp(y_hat)
    y_hat = pd.DataFrame(data=y_hat)
    return y_hat



###############
## read in data
X = pd.read_csv('train.csv',sep=',',names=['timestamp','A','B','C','D','E','F'])
y = pd.read_csv('train_y.csv',names='y')

## create matrix for regression
X = create_design_matrix(X)
#print list(X.columns)

## split data in train and test set
X_tr, X_te, y_tr, y_te = skcv.train_test_split(X,y,train_size=0.75)
## make them dataframes
X_tr = pd.DataFrame(data=X_tr,columns=X.columns)
X_te = pd.DataFrame(data=X_te,columns=X.columns)
y_tr = pd.DataFrame(data=y_tr)
y_te = pd.DataFrame(data=y_te)
## log-transformation of training response var
## y_te is not transformed
y_tr = np.log(y_tr)
#print y_tr
## test how good is the prediction on validation set
print '\n**Regression on test set**\n'
#print 'Score on validation set: ', kernelregression(X_tr,y_tr,X_te,y_te)


print '\n\n**Train and Test for validation set**\n\n'



####################
#### Validation Area


X_val = pd.read_csv('validate.csv',sep=',',names=['timestamp','A','B','C','D','E','F'])
X_val = create_design_matrix(X_val)
######
#print X_val.columns
## make log trabsformation on response
y = np.log(y)
## use full X matrix for training
y_hat = kernelregression1(X,y,X_val)
y_hat.to_csv('out_rbf9.csv', index=False,header=False)
print 'prediction finished'
