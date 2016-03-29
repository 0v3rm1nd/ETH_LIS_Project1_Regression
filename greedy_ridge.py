import pandas as pd
import numpy as np
import sklearn.linear_model as sklin
import sklearn.cross_validation as skcv
import sklearn.grid_search as skgs
import sklearn.metrics as skmet
import math
import datetime

## You have to install the pandas modul

## Overview
## fct: ret_time_values
##      extracts vars from given timestamp
##      define manually new features
## fct: create_design_matrix
##      puts the defined vars in one dataframe
##      define manually new features
## fct: logscore
##      evaluation function on validation set
## fct: ridgeregression
##      regression function
## fct: get_scores
##      calculation of scores each run of greedy
## fct: get_greedy_scores
##      construction of greedy matrix and greedy scores



##############
## functions
def ret_time_values(timestamp):
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
    second = pd.DatetimeIndex(timestamp).second
    day_int = pd.DatetimeIndex(timestamp).day
    ret = year
    ret[list(month.columns)] = month
    ret[list(day.columns)] = day
    ret[list(hour_d.columns)] = hour_d
    ret['hour'] = hour
    ret['hour_exp'] = np.exp(hour)
    #ret['hour_log'] = np.log(hour)
    ret['hour_^2'] = hour**2
    ret['hour_^3'] = hour**3
    ret['hour_^4'] = hour**4
    ret['hour_^5'] = hour**5
    ret['hour_^6'] = hour**6
    ret['hour_^7'] = hour**7
    ret['hour_^8'] = hour**8
    ret['hour_^9'] = hour**9
    ret['hour_^10'] = hour**10
    ret['day_int'] = day_int
    ret['day*hour'] = hour * day_int
    ret['minute'] = minute
    ret['minute_exp'] = np.exp(minute)
    ret['minute_^2'] = minute**2
    return ret

def create_design_matrix(V):
    X = V
    timestamp = pd.to_datetime(X['timestamp'],format='%Y-%m-%d %H:%M:%S')
    time_splitted = ret_time_values(timestamp)
    
    B = pd.get_dummies(X['B'])
    B.columns = ['B1','B2','B3','B4']

    del X['timestamp']
    del X['B']

    X['D_log'] = np.log(X['D'])
    
    ## center matrix
    for i in list(X.columns):
        X[i] = (X[i] - np.mean(X[i])) / np.std(X[i])

    for i in list(X.columns):
        X[str(i + '_exp')] = np.exp(X[i])
        X[str(i + '_^2')] = X[i]*2
        X[str(i + '_^3')] = X[i]*3
        X[str(i + '_^4')] = X[i]*4
        X[str(i + '_^5')] = X[i]*5
        #X[str(i + '_log')] = np.log(X[i])
    
    X['AC'] = X['A']*X['C']
    X['AD'] = X['A']*X['D']
    X['AE'] = X['A']*X['E']
    X['AF'] = X['A']*X['F']
    X['CD'] = X['C']*X['D']
    X['CE'] = X['C']*X['E']
    X['CF'] = X['C']*X['F']
    X['DE'] = X['D']*X['E']
    X['DF'] = X['D']*X['F']
    X['EF'] = X['E']*X['F']

    

    X['ACD'] = X['A']*X['C']*X['D']
    X['ACE'] = X['A']*X['C']*X['E']
    X['ACF'] = X['A']*X['C']*X['F']
    X['CDE'] = X['C']*X['D']*X['E']
    X['CDF'] = X['C']*X['D']*X['F']
    X['DEF'] = X['D']*X['E']*X['F']

    X['AC'] = np.exp(X['A']*X['C'])
    X['AD'] = np.exp(X['A']*X['D'])
    X['AE'] = np.exp(X['A']*X['E'])
    X['AF'] = np.exp(X['A']*X['F'])
    X['CD'] = np.exp(X['C']*X['D'])
    X['CE'] = np.exp(X['C']*X['E'])
    X['CF'] = np.exp(X['C']*X['F'])
    X['DE'] = np.exp(X['D']*X['E'])
    X['DF'] = np.exp(X['D']*X['F'])
    X['EF'] = np.exp(X['E']*X['F'])

    X['ACD'] = np.exp(X['A']*X['C']*X['D'])
    X['ACE'] = np.exp(X['A']*X['C']*X['E'])
    X['ACF'] = np.exp(X['A']*X['C']*X['F'])
    X['CDE'] = np.exp(X['C']*X['D']*X['E'])
    X['CDF'] = np.exp(X['C']*X['D']*X['F'])
    X['DEF'] = np.exp(X['D']*X['E']*X['F'])

    X[list(B.columns)] = B
    X[list(time_splitted.columns)] = time_splitted

    X['sumRvars'] = X['A']+X['C']+X['D']+X['E']+X['F']
    X['hour+A'] = X['hour'] + X['A']
    X['exp(hour + A)'] = np.exp(X['hour'] + X['A'])
    X['sumAC'] = X['A']+X['C']
    X['sumCD'] = X['C']+X['D']
    X['sumDE'] = X['D']+X['E']
    X['sumEF'] = X['E']+X['F']
    X['hour+D'] = X['hour'] + X['D']
    X['hour+E'] = X['hour'] + X['E']
    X['hour+F'] = X['hour'] + X['F']
    X['hour*D'] = X['hour'] * X['D']

    X['hour*A'] = X['hour']*X['A']
    X['hour+A+C+D+E+F'] = X['hour']+X['A']+X['C']+X['D']+X['E']+X['F']
    X['hour+sqrt(A)'] = X['hour'] + np.power(X['A'],1/2)
    X['hour+5root(A)'] = X['hour'] + np.power(X['A'],1/5)
    X['hour_centr'] = (X['hour'] - np.mean(X['hour'])) / np.std(X['hour'])
    X['hourA'] = X['hour']*X['A']
    #X['hourB'] = X['hour']*X['B']
    X['hourC'] = X['hour']*X['C']
    X['hourD'] = X['hour']*X['D']
    X['hourE'] = X['hour']*X['E']
    X['hourF'] = X['hour']*X['F']
    return X


def logscore(y, y_hat):
    """
    y: Evaluation response
    y_hat: predicted response from regression
    """
    y_hat = np.clip(y_hat,0,np.inf)
    logdif = np.log(1 + y) - np.log(1 + y_hat)
    return np.sqrt(np.mean(np.square(logdif)))


def ridgeregression(X_tmp,y_tr,X_te_tmp,y_te):
    """
    X_tmp: Train matrix
    y_tr: Train Response var
    X_te_tmp: valuation matrix
    y_te: Response valuation var
    """
    regressor_ridge = sklin.Ridge()
    param_grid = {'alpha':np.linspace(0,10,20)}
    neg_scorefun = skmet.make_scorer(lambda x,y: -logscore(x,y))
    grid_search = skgs.GridSearchCV(regressor_ridge,param_grid,scoring=neg_scorefun,cv=5)
    grid_search.fit(X_tmp,y_tr)
    best = grid_search.best_estimator_
    y_hat = best.predict(X_te_tmp)
    y_hat = np.exp(y_hat)
    return logscore(y_te,y_hat).values 


def get_scores(X_tmp,X_te_tmp,X_t,y_tr,X_te,y_te):
    """
    X_tmp: greedy train matrix
    X_te_tmp: greedy valuation matrix
    X_t: greedy train input matrix
    y_tr: Train Response var
    X_te: greedy valuation input matrix
    y_te: Valuation Response var
    """
    scores = []
    # loop over all columns from greedy input matrix
    # return score on regression on each individual column
    for i in X_t.columns:
        X_tmp[i] = X_t[i]
        X_te_tmp[i] = X_te[i]
        print i
        scores.append([ridgeregression(X_tmp,y_tr,X_te_tmp,y_te),i])
        del X_tmp[i]
        del X_te_tmp[i]
    return scores

def get_greedy_scores(X,y,k):
    """
    X: Data Matrix
    y: Response variable
    """
    # X_tmp greedy train matrix
    X_tmp = pd.DataFrame()
    # X_te_tmp: greedy validation matrix
    X_te_tmp = pd.DataFrame()
    # greedy scores
    scores = []
    #split data in train and test
    X_tr, X_te, y_tr, y_te = skcv.train_test_split(X,y,train_size=0.75)
    #make them dataframes
    X_tr = pd.DataFrame(data=X_tr,columns=X.columns)
    X_te = pd.DataFrame(data=X_te,columns=X.columns)
    y_tr = pd.DataFrame(data=y_tr)
    y_te = pd.DataFrame(data=y_te)
    # log transformation on y_tr
    y_tr = np.log(y_tr)
    for i in range(k):
        print '\n\n** run : **',i+1
        # get scores of regression
        # var-name 'index' is not a good choice...left it
        index = get_scores(X_tmp,X_te_tmp,X_tr,y_tr,X_te,y_te)
        # store best result
        scores.append(index[index.index(min(index))])
        # extract column-name and score from best result
        tmp = index[index.index(min(index))][1]
        score = index[index.index(min(index))][0]
        # add best column to greedy matrix
        X_tmp[tmp] = X_tr[tmp]
        # add best column to greedy validation matrix
        X_te_tmp[tmp] = X_te[tmp]
        # drop this column in both matrices X_tr, X_te
        del X_tr[tmp]
        del X_te[tmp]
        print '\n\n**Selected: **',tmp
        print '\n**Score: **', score
    
    #return X_tmp with columns picked by greedy
    #return matrix X_tr with columns not picked by greedy
    #return the greedy scores
    return X_tmp,X_tr, scores





###############
###############
## read in data
X = pd.read_csv('train.csv',sep=',',names=['timestamp','A','B','C','D','E','F'])
y = pd.read_csv('train_y.csv',names='y')

X = create_design_matrix(X)

## define how much vars you wanna receive from greedy fw selection
k = 1

#################
## greedy_X contains greedy columns for k vars
## greedy_X_rest is uninteresting, keeps the rest of the vars not picked by greedy...left it
## greedy_scores: performance result
#################
greedy_X, greedy_X_rest, greedy_scores = get_greedy_scores(X,y,k)

print '\n\n**greedy finished**\n'
#greedy_X.to_csv('greedy_X.csv',index=False,header=True)
#greedy_X_rest.to_csv('greedy_X_rest.csv',index=False,header=True)
greedy_scores = pd.DataFrame(data=greedy_scores)
greedy_scores.to_csv('greedy_scores_new29.csv',index=False,header=False)
print 'greedy scores stored\n\n'



########################
########################
########################
## Validation Area

## Extract greedy picked columns for validation
V = pd.DataFrame()
V = X[list(greedy_X.columns)]
# here we need the log transformed y
y_v = np.log(y)
## Train with full Matrix V

################
## Grid Search
regressor_ridge = sklin.Ridge()
param_grid = {'alpha':np.linspace(0,10,20)}
neg_scorefun = skmet.make_scorer(lambda x,y: -logscore(x,y))
grid_search = skgs.GridSearchCV(regressor_ridge,param_grid,scoring=neg_scorefun,cv=5)
grid_search.fit(V,y_v)

best = grid_search.best_estimator_
## best score here is presented on log(y)
print 'best score = ', -grid_search.best_score_


##################
## Validation
X_val = pd.read_csv('validate.csv',sep=',',names=['timestamp','A','B','C','D','E','F'])
X_val = create_design_matrix(X_val)
######

##select greedy picked columns for prediction
V_val = X_val[list(greedy_X.columns)]

####
####
Ypred = best.predict(V_val)
Ypred = np.exp(Ypred)
Ypred = pd.DataFrame(data=Ypred)
#Ypred.to_csv('out102.csv',index=False,header=False)
print 'prediction finished'
