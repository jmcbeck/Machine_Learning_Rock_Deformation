#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:49:24 2022

@author: mcbeck
"""
import statistics as stat
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import shap
import pickle

import random
from datetime import datetime
random.seed()

fout = 'C:/Users/jessicmc/research/machine_learning/output/'
fold = 'C:/Users/jessicmc/research/dvc/matlab/txt/'

# data from make_balanced_data.m

# test for all experiments
exps = ['DD04', 'DD05_05', 'DD_10', 'DD_11', 'DD_12', 'DD_13']
#exps = ['DD04'] # for testing

#expstr = 'allexp'

# groups of expeirments with similar confining stress?

# effect of time slicing is probably stronger for 
# models trained on individual experiments
#exps = ['DD04']
#expstr = 'DD04'

# train on individaul experiments, see which ones perform worse
# then potentially remove from grouping of all the experiments

#expstrs = ['DD04', 'DD05_05', 'DD_10', 'DD_11', 'DD_12', 'DD_13']
#expstrs = ['DD_10', 'DD_11', 'DD_12', 'DD_13']
#expstrs = ['DD_13']
#expstr = 'allexps' # rerun for random with multiple iterations
#exprstrs = ['DD05_05', 'DD_11']
#exprstrs = ['DD04', 'DD_10', 'DD_12', 'DD_13']
#exprstrs = ['DD04', 'DD05_05', 'DD_10', 'DD_11', 'DD_12', 'DD_13']
exprstrs = ['DD05_05', 'DD_10', 'DD_11', 'DD_12', 'DD_13', 'DD04']
#exprstrs = ['DD_12', 'DD_13']


# indiviadual experiments, smaller time slices
# pin point distance from failure when the accuracy is above 0.7 e.g.


# loop of individual experiments, and combined experiments
# first check if possible to get high accuracy with combined experiments
# more meaningful to group together
# difficult to interpret differences between experiments

# randomize splitting the training and testing
#split_str = 'rand' # likely higher scores
#split_str = 'time' # likely lower scores

#splits = ['rand', 'time']
#splits = ['rand']
splits = ['time']
#splits = ['expDD04', 'expDD05_05', 'expDD_10', 'expDD_11', 'expDD_12', 'expDD_13']

# predict locations where there is high volumetric strain and shear strain
pred_str = 'ishighIJ'

instr = 'morefts'

for expstr in exprstrs:

    # if not combining all the experiments together, then only use the one experiment listed
    if 'all' not in expstr:
        exps = [expstr]

    fti=2
    while fti<=2:
        #  leave off the coordinates, and ti
        if fti==1:
        
            ft_sub = 'nocoor'
            #feats = ['nep', 'nsig', 'I1', 'J2', 'volmn1', 'volstdn1', 'volp25n1', 'volp50n1', 'volp75n1', 'shrmn1', 'shrstdn1', 'shrp25n1', 'shrp50n1', 'shrp75n1', 'volmn2', 'volstdn2', 'volp25n2', 'volp50n2', 'volp75n2', 'shrmn2', 'shrstdn2', 'shrp25n2', 'shrp50n2', 'shrp75n2']
            feats = ['nep', 'nsig', 'gI1mean', 'gI1std', 'gI1min', 'gI1p25', 'gI1p50', 'gI1p75', 'gI1p90', 'gI1max', 'gJ2mean', 'gJ2std', 'gJ2min', 'gJ2p25', 'gJ2p50', 'gJ2p75', 'gJ2p90', 'gmaxJ2', 'I1', 'J2', 'n1I1mean', 'n1I1std', 'n1I1min', 'n1I1p25', 'n1I1p50', 'n1I1p75', 'n1I1p90', 'n1I1max', 'n1J2mean', 'n1J2std', 'n1J2min', 'n1J2p25', 'n1J2p50', 'n1J2p75', 'n1J2p90', 'n1maxJ2', 'n2I1mean', 'n2I1std', 'n2I1min', 'n2I1p25', 'n2I1p50', 'n2I1p75', 'n2I1p90', 'n2I1max', 'n2J2mean', 'n2J2std', 'n2J2min', 'n2J2p25', 'n2J2p50', 'n2J2p75', 'n2J2p90', 'n2maxJ2', 'n3I1mean', 'n3I1std', 'n3I1min', 'n3I1p25', 'n3I1p50', 'n3I1p75', 'n3I1p90', 'n3I1max', 'n3J2mean', 'n3J2std', 'n3J2min', 'n3J2p25', 'n3J2p50', 'n3J2p75', 'n3J2p90', 'n3maxJ2']
    
        # remove the normalized ep and sigd
        elif fti==2:
            ft_sub = 'nocoortime'
            feats = ['gI1mean', 'gI1std', 'gI1min', 'gI1p25', 'gI1p50', 'gI1p75', 'gI1p90', 'gI1max', 'gJ2mean', 'gJ2std', 'gJ2min', 'gJ2p25', 'gJ2p50', 'gJ2p75', 'gJ2p90', 'gmaxJ2', 'I1', 'J2', 'n1I1mean', 'n1I1std', 'n1I1min', 'n1I1p25', 'n1I1p50', 'n1I1p75', 'n1I1p90', 'n1I1max', 'n1J2mean', 'n1J2std', 'n1J2min', 'n1J2p25', 'n1J2p50', 'n1J2p75', 'n1J2p90', 'n1maxJ2', 'n2I1mean', 'n2I1std', 'n2I1min', 'n2I1p25', 'n2I1p50', 'n2I1p75', 'n2I1p90', 'n2I1max', 'n2J2mean', 'n2J2std', 'n2J2min', 'n2J2p25', 'n2J2p50', 'n2J2p75', 'n2J2p90', 'n2maxJ2', 'n3I1mean', 'n3I1std', 'n3I1min', 'n3I1p25', 'n3I1p50', 'n3I1p75', 'n3I1p90', 'n3I1max', 'n3J2mean', 'n3J2std', 'n3J2min', 'n3J2p25', 'n3J2p50', 'n3J2p75', 'n3J2p90', 'n3maxJ2']
        elif fti==3:
            ft_sub = 'global'
            feats = ['I1', 'J2', 'gI1mean', 'gI1std', 'gI1min', 'gI1p25', 'gI1p50', 'gI1p75', 'gI1p90', 'gI1max', 'gJ2mean', 'gJ2std', 'gJ2min', 'gJ2p25', 'gJ2p50', 'gJ2p75', 'gJ2p90', 'gmaxJ2']
        elif fti==4:
            ft_sub = 'n1'
            feats = ['I1', 'J2', 'n1I1mean', 'n1I1std', 'n1I1min', 'n1I1p25', 'n1I1p50', 'n1I1p75', 'n1I1p90', 'n1I1max', 'n1J2mean', 'n1J2std', 'n1J2min', 'n1J2p25', 'n1J2p50', 'n1J2p75', 'n1J2p90', 'n1maxJ2']
        elif fti==5:
            ft_sub = 'n1n2'
            feats = ['I1', 'J2', 'n1I1mean', 'n1I1std', 'n1I1min', 'n1I1p25', 'n1I1p50', 'n1I1p75', 'n1I1p90', 'n1I1max', 'n1J2mean', 'n1J2std', 'n1J2min', 'n1J2p25', 'n1J2p50', 'n1J2p75', 'n1J2p90', 'n1maxJ2', 'n2I1mean', 'n2I1std', 'n2I1min', 'n2I1p25', 'n2I1p50', 'n2I1p75', 'n2I1p90', 'n2I1max', 'n2J2mean', 'n2J2std', 'n2J2min', 'n2J2p25', 'n2J2p50', 'n2J2p75', 'n2J2p90', 'n2maxJ2']
        elif fti==6:
            ft_sub = 'n1n2n3'
            feats = ['I1', 'J2', 'n1I1mean', 'n1I1std', 'n1I1min', 'n1I1p25', 'n1I1p50', 'n1I1p75', 'n1I1p90', 'n1I1max', 'n1J2mean', 'n1J2std', 'n1J2min', 'n1J2p25', 'n1J2p50', 'n1J2p75', 'n1J2p90', 'n1maxJ2', 'n2I1mean', 'n2I1std', 'n2I1min', 'n2I1p25', 'n2I1p50', 'n2I1p75', 'n2I1p90', 'n2I1max', 'n2J2mean', 'n2J2std', 'n2J2min', 'n2J2p25', 'n2J2p50', 'n2J2p75', 'n2J2p90', 'n2maxJ2', 'n3I1mean', 'n3I1std', 'n3I1min', 'n3I1p25', 'n3I1p50', 'n3I1p75', 'n3I1p90', 'n3I1max', 'n3J2mean', 'n3J2std', 'n3J2min', 'n3J2p25', 'n3J2p50', 'n3J2p75', 'n3J2p90', 'n3maxJ2']
        elif fti==7:
            ft_sub = 'fttop1'
            feats = ['I1', 'J2', 'gI1mean', 'gI1std']
        elif fti==8:
            ft_sub = 'fttop2'
            feats = ['I1', 'J2', 'gI1mean', 'gI1std', 'gJ2mean']
    
        elif fti==9:
            ft_sub = 'global100'
            feats = ['I1', 'J2', 'g100I1mean', 'g100I1std', 'g100J2mean', 'g100J2std']
            instr = 'ftglobal'
        elif fti==10:
            ft_sub = 'global50'
            feats = ['I1', 'J2', 'g50I1mean', 'g50I1std', 'g50J2mean', 'g50J2std']
            instr = 'ftglobal'
        elif fti==11:
            ft_sub = 'global25'
            feats = ['I1', 'J2', 'g25I1mean', 'g25I1std', 'g25J2mean', 'g25J2std']
            instr = 'ftglobal'
        elif fti==12:
            ft_sub = 'global10'
            feats = ['I1', 'J2', 'g10I1mean', 'g10I1std', 'g10J2mean', 'g10J2std']
            instr = 'ftglobal'
          
    
    
        numf = len(feats)
     
        for split_str in splits:
            
       
            props = expstr+'_'+ft_sub+'_'+pred_str+'_'+split_str
                
            fscore = fout+props+'_'+instr+'_score.txt'
            fimp = fout+props+'_'+instr+'_shap.txt'
            
            print(fscore)
            
            score_str = "nepmin nepmax it num_train num_test recall precision f1 accuracy \n";
            imp_str = "feat_str nepmin nepmax it feat_num feat_shap \n"
            
            # limit the data to range of normalized ep
            nii=1
            while nii<=6:
            
                if nii==1:
                    nepmin = 0
                    nepmax = 0.5  
                elif nii==2:
                    nepmin = 0.5
                    nepmax = 0.6  
                elif nii==3:
                    nepmin = 0.6
                    nepmax = 0.7  
                elif nii==4:
                    nepmin = 0.7
                    nepmax = 0.8
                elif nii==5:
                    nepmin = 0.8
                    nepmax = 0.9  
                elif nii==6:
                    nepmin = 0.8
                    nepmax = 1.0   
                else:
                    nepmin = 0
                    nepmax = 1.0   
                        
                nestr = 'nep'+str(nepmin)+'_'+str(nepmax)
                
                # combine all the experiments together
                df = pd.DataFrame()
                
                for exp in exps:
                    txtfile = exp+'_strains_high_'+instr+'_p80.txt'
                    fname = fold+txtfile
                    
                    df_sm = pd.read_csv(fname, delim_whitespace=True)  
                    df_sm = df_sm.dropna().copy()
                    #df_sm['exp'] = exp
                    
                    nd = len(df_sm)
                    vals = [exp]*nd
                    
                    df_sm.insert(0, "exp", vals, True)
                    
                    df = pd.concat([df, df_sm], ignore_index=True)
                
                neps = df['nep'].values
                nepu = set(df['nep'].values)
                print('normalized eps in full dataset')
                print(sorted(nepu))
                
                indsmin = neps<=nepmin
                df = df.drop(df[indsmin].index)
                
                neps = df['nep'].values
                indsmax = neps>=nepmax
                df = df.drop(df[indsmax].index)
                
                neps = set(df['nep'].values)
                print('normalized eps in remaining dataset, nep=', nepmin, '-', nepmax)
                print(sorted(neps))
                
                if len(neps)>0:
                    df_scale = df.copy()
                
                    trans = RobustScaler()
                    df_scale[feats] = trans.fit_transform(df_scale[feats].values)  
                    pred = df_scale[pred_str].values
                    
                    if 'time' in split_str:
                        tmax=10
                    elif 'exp' in split_str:
                        tmax=1
                    else:
                        tmax=5
                    
                    it=0
                    while it<tmax:
                            
                        if 'exp' in split_str:
                            exptr = split_str[3:]
                            
                            inds = (df_scale['exp']==exptr)==False
                            
                        elif 'time' in split_str:
                            # randomly select training and testing so that 
                            # one time (scan) is in either training or testing
                            times = df_scale['nep'].values
                            timesu = set(times)
                            indt = np.random.uniform(0, 1, len(timesu)) <= .60
                            tri =0
                            while (sum(indt)==0 or sum(indt)==len(indt)) and tri<50:
                                indt = np.random.uniform(0, 1, len(timesu)) <= .60
                                tri=tri+1
                            
                            inds = [0 if e==1 else 0 for i, e in enumerate(times)]
                            ti=0
                            for time in times:
                                loc = list(timesu).index(time)
                                ib = indt[loc]
                                inds[ti] = ib
                                ti=ti+1
                                        
                        elif 'rand' in split_str:
                            # split randomly
                            inds = np.random.uniform(0, 1, len(df_scale)) <= .70 
                
                        df_scale['is_train'] = inds
                        df['is_train'] = inds
                        trainun, testun = df[df['is_train']==True], df[df['is_train']==False]
                
                        print('splitting: ', split_str)
                
                        # y_tr = trainun[pred_str].values
                        s_tr = set(trainun['nep'].values)
                        print('nep in the training data')
                        print(sorted(s_tr))      
                        
                        s_te = set(testun['nep'].values)
                        print('nep in the testing data')
                        print(sorted(s_te))       
                        
                        exp_tr = set(trainun['exp'].values)
                        print('experiments in the training data')
                        print(sorted(exp_tr)) 
                        
                        exp_te = set(testun['exp'].values)
                        print('experiments in the testing data')
                        print(sorted(exp_te))  
                        
                        train, test = df_scale[df_scale['is_train']==True], df_scale[df_scale['is_train']==False]
                
                        x_train = train[feats]
                        y_train = train[pred_str]
                          
                        x_test = test[feats]
                        y_test = test[pred_str]
                        
                        X = df_scale[feats]
                        y = df_scale[pred_str]
                        
                        n_train = len(y_train)
                        n_test = len(y_test)
                        rat = n_test/(n_train+n_test)
                        print('# train, # test, test/(test+train):', n_train, n_test, round(rat, 2))
                        
                        
                        current_time = datetime.now()
                        hour = current_time.time()
                        print('time starting training model:')
                        print(hour)
                        # remember the data used for testing to see if
                        # certain scans lead to bad scores
                        
                        # weighted features
                        # add features to explain subsets of data
                        data_dmatrix = xgb.DMatrix(data=X,label=y)
                        
                        # max_depth: Maximum depth of a tree. 
                        # Increasing this value will make the model more complex and more likely to overfit
                        # colsample_bytree: subsample ratio of columns when constructing each tree
                        # alpha: L1 regularization term on weights. Default=0
                        # Increasing this value will make model more conservative
                        # learning_rate, eta = Typical final values to be used: 0.01-0.2
                        #xgb_clf = xgb.XGBRegressor(objective ='reg:squarederror')
                        
                        xgb_clf = xgb.XGBClassifier()
                        
                        # parameters = {'colsample_bytree':[0.7, 0.8, 0.9], 'alpha':[0, 3, 5], 
                        #        'learning_rate': [0.1, 0.2, 0.3],
                        #        'n_estimators': [100, 200, 300], 'max_depth':[3, 4, 5, 6]}
                        parameters = {'colsample_bytree':[0.7, 0.8], 'alpha':[0, 3], 
                                'learning_rate': [0.1, 0.2],
                                'n_estimators': [100, 200], 'max_depth': [3, 4, 5]}
                        #parameters = {'colsample_bytree':[0.7], 'alpha':[3], 
                        #      'learning_rate': [0.1],
                        #      'n_estimators': [100], 'max_depth':[4]}
                        grid_search = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=10, n_jobs=-1)
                        
                        grid_search.fit(x_train, y_train)
                        xg_reg = grid_search.best_estimator_
                        
                        preds = xg_reg.predict(x_test)
                        preds_train = xg_reg.predict(x_train)
                        
                        # rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
                        # rmse_test = np.sqrt(mean_squared_error(y_test, preds))
                        
                        # r2_test = r2_score(y_test, preds)
                        # r2_train = r2_score(y_train, preds_train)
                        
                        accuracy = accuracy_score(y_test, preds)
                        recall = recall_score(y_test, preds)
                        precision = precision_score(y_test, preds)
                        f1 = f1_score(y_test, preds)
                
                        current_time = datetime.now()
                        hour = current_time.time()
                        print('time finished training model:')
                        print(hour)
                
                        #curr_str = "%.0f %.0f %.0f %.5f %.5f %.5f %.5f \n" % (it, n_train, n_test, rmse_train, r2_train, rmse_test, r2_test)
                        curr_str = "%.2f %.2f %d %.0f %.0f %.5f %.5f %.5f %.5f \n" % (nepmin, nepmax, it, n_train, n_test, recall, precision, f1, accuracy)
                        
                        score_str = score_str+curr_str
                        print(score_str)
                        
                        
                        shap_vals = shap.TreeExplainer(xg_reg).shap_values(train[feats], check_additivity=False)
                        shap_comb = shap_vals.transpose()
                        
                        shap_mean = []
                        num_f = len(shap_comb)
                        for fi in range(len(shap_comb)):              
                            vabs = abs(shap_comb[fi])
                            v_mean = stat.mean(vabs)       
                            shap_mean.append(v_mean)
                    
                        shapl = list(zip(feats, shap_mean, range(1, numf+1)))
                        shapl.sort(key=lambda tup: tup[2], reverse=False)
                            
                        shps = ""
                        for ft in shapl:
                            #ft[0]+" "+str(nx)+" "+str(it)+" "+str(ft[2])+" "+str(round(ft[1], 4))+"\n"
                            shp_str = "%s %.2f %.2f %d %.5f %.5f \n" % (ft[0], nepmin, nepmax, it, ft[2], round(ft[1], 4))
                            shps = shps+shp_str
                        
                
                
                        imp_str = imp_str+shps
                
                
                        shapl.sort(key=lambda tup: tup[1], reverse=True)
                            
                        shps = ""
                        for ft in shapl:
                            #ft[0]+" "+str(nx)+" "+str(it)+" "+str(ft[2])+" "+str(round(ft[1], 4))+"\n"
                            shp_str = "%s %d %.5f \n" % (ft[0], ft[2], round(ft[1], 4))
                            shps = shps+shp_str
                
                        print(shps)
                
                        f= open(fscore, "w")
                        f.write(score_str)
                        f.close()
                        
                        f= open(fimp, "w")
                        f.write(imp_str)
                        f.close()
                        print(fimp)
    
                        mname = fout+'models/'+props+'_it'+str(it)+'.pkl'
                        with open(mname,'wb') as f:
                            pickle.dump(xg_reg,f)
                        
      
                        it=it+1
                        
                
                        
                nii=nii+1
        fti=fti+1
                        
                    
                    
