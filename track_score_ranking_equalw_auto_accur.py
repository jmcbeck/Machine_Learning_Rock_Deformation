# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

#filen = 'txts/M8_2_track_params_cont.txt' 
#filen = 'txts/M8_track_params_cont.txt'  
#filen = 'txts/MONZ5_track_params_cont.txt' 
filen = 'txts/WG04_track_params_cont.txt' 

itr = 5000 # number of iterations in solver
tsi = 0.20 # test size

scale = 1 # scale features
col_scale = ['min-dist', 'mean-dist',
      'median-dist', 'mean-10_dist', 'median-10_dist', 'theta1', 'theta2',
       'theta3', 'd1', 'd2', 'd3', 'aniso', 'elong', 'volume']

# percentile of volume range
# only from 10-100th percentile
q=10
q_next = 100
    
# change in volume percentage
#perc = 0.70;
is_equal=1; # do or do not down sample to make equal categories
#percs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
percs = [0.5]
for perc in percs:
    print(perc)
    txt_name = 'txts/score_EXP_p'+str(round(perc*100))+'_p'+str(q)+'_wEX_accur.txt'
    exp_str = filen.split('_')
    exp_str = exp_str[0]
    exp_s = exp_str[5:]
    if 'M8_2' in filen:
        exp_s = exp_s+'_2'
        
    txt_name = txt_name.replace('EXP', exp_s)
    
    # run model multiple times for different fracture populations
    
    recalls = []
    accurs = []
    precis = []
    rocs = []
    num_g1 = []
    num_g0 = []
    
    ranks = []
    rankscv = []
    qfeats = []
    witers = []
    featns = []
    coef_tot = []
    sig_tot = []
    scorcv = []
    selcv = []
    coeffs = []
    #statf = 'p tp tn fp fn TPR FPR \n'
    
    it=0;
    maxi = 50;
    while it<maxi:
        # input the data into a data frame
        df = pd.read_csv(filen, delim_whitespace=True)  
        
        # remove fractures with volume outside percentile range 
        df_thresh = df.dropna().copy()
        sigs = df_thresh['sig']
        
        i_c=0
        remn=0
        inds = []
        sig_uni = list(set(sigs))
        for si in sig_uni:
            df_sig = df_thresh[df_thresh["sig"]==si]
            df_vol = df_sig["volume"].values
            
                       
            thr_low = np.percentile(df_vol, q) 
            thr_high = np.percentile(df_vol, q_next)
            
            # remove these boolean values
            # list of indices of this time
            bool_1 = df_vol<thr_low
            bool_2 = df_vol>thr_high
            
            #ind_rem = list(compress(list_a, not(fil)))
            i=0 
            for bol1 in bool_1:
                bol2 = bool_2[i]
                if bol1 or bol2:
                    remn= remn+1
                    inds.append(i+i_c)
                    
                i=i+1
                
            i_c=i_c+i
        
        df_thresh = df_thresh.drop(df_thresh.index[inds])
        df = df_thresh
        
        #q_prev = q
        # only consider fractures growing by X% of their volume
        # within certain %s
        df = df[abs(df["delvol"])> perc*df["volume"]]
        
        # count the number of true and faulse
        grow_bin = df["is_grow"].values
        num_1 = len(grow_bin[grow_bin==1])
        num_0 = len(grow_bin[grow_bin==0])
        
        # randomly select subsets of growing fractures so ratio of num_0 and num_1= 1
        # find the index of values to drop, 
        if is_equal and num_0<num_1:
            #ind_drop = []
            df_wei = df.copy()
            growing = df_wei[df_wei["is_grow"].values==1]  
            ind_gr = growing.index.values
            
            # select which indexes of growing to drop
            dropn = num_1-num_0
            dri=0
            
            while dri<dropn:
                # select index in ind_gr to drop, remove from list of ind_gr
                
                ind_dr_c = random.randrange(len(ind_gr))
                row = ind_gr[ind_dr_c]
                ind_gr = np.delete(ind_gr, ind_dr_c)  
        
                df_wei = df_wei[df_wei.index.values != row]
                #ind_drop.append(row)
                dri=dri+1
        
            df = df_wei
        
        grow_bin = df["is_grow"].values
        num_1 = len(grow_bin[grow_bin==1])
        num_0 = len(grow_bin[grow_bin==0])
    
        # scale the data    
        df_scale = df.copy()
        trans = RobustScaler()# works better with outliers
        df_scale[col_scale] = trans.fit_transform(df[col_scale].values)  
        df = df_scale
        
        
        no_nan_data = df.dropna().copy()
        train_x, test_x, train_y, test_y = train_test_split(no_nan_data[col_scale], no_nan_data['is_grow'], test_size=tsi, random_state=0)
        
        # parameters of logistic regression
        # Dictionary with parameters names (string) as keys and lists of parameter settings to try as values
        params = {
            'C':[10., 1., 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
            ,'penalty':['l1', 'l2']
            ,'solver':['liblinear'] 
        }
        
        # This class implements regularized logistic regression using the ‘liblinear’ library, 
        # ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers. 
        # It can handle both dense and sparse input. 
        lr = LogisticRegression(max_iter=itr, class_weight='balanced') # influence?
        
        # Exhaustive search over specified parameter values for an estimator.
        # The parameters of the estimator used to apply these methods are optimized 
        # by cross-validated grid-search over a parameter grid.
        # lr= estimator object
        # params of ML
        # n_jobs = how many processors to use
        # cv= Determines the cross-validation splitting strategy, with integer: number of 
        grid_search = GridSearchCV(lr, param_grid=params, n_jobs=-1, cv=5)
        # fit to training data
        grid_search.fit(train_x, train_y)
        # best_estimator_ : estimator or dict:
        
        # Estimator that was chosen by the search, i.e. estimator which 
        # gave highest score (or smallest loss if specified) on the left out data.
        clf = grid_search.best_estimator_
        
        # calculate confusion matrix
        # top row= correct predictions
        # bottom row= incorrect predictions
        predictions = clf.predict(test_x)
        cm = metrics.confusion_matrix(test_y, predictions)
        
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        
        predict_prob = clf.predict_proba(test_x)   
        rocdf = pd.DataFrame({'test_true':test_y})
        rocdf['test_pred'] = predictions       
        fpr, tpr, thresholds = metrics.roc_curve(rocdf['test_true'], predict_prob[:,1])
        roc_auc = metrics.roc_auc_score(rocdf['test_true'], predict_prob[:,1])
           
        accur = (tp+tn)/(tp+tn+fp+fn)
        recall = tp/(tp+fn)
        preci = tp/(tp+fp)
        
        accurs.append(accur)
        recalls.append(recall)
        rocs.append(roc_auc)
        precis.append(preci)
        
        # do recurrsive feature elimination to rank features    
        rfe = RFE(clf, 1)
        select = rfe.fit(train_x, train_y) 
        rank = select.ranking_
        sup = select.support_
          
        # recurr feature elinaiton with cross val
        rfecv = RFECV(clf, step=1, cv=3, scoring='accuracy')
        fitcv = rfecv.fit(train_x, train_y)
        rankcv = fitcv.ranking_
        supcv = fitcv.support_
        sco = fitcv.grid_scores_ # how to link to correct features...
        
        dat = clf.coef_.transpose()
        dat = [abs(x) for x in dat]
        coef_df = pd.DataFrame(dat, index=col_scale, columns=['coefs'])
        
        coef_str = coef_df.to_csv(sep=' ', index=False, header=True)
        coef_li = coef_str.split()
            
        fi=0
        for r in rank:
            featns.append(fi)
            q_r = str(q)+'_'+str(q_next)
            qfeats.append(q_r)
            witers.append(it)
            ranks.append(r)
            rankscv.append(rankcv[fi])
            scorcv.append(sco[fi])
            selcv.append(supcv[fi])
            coeffs.append(coef_li[fi+1])
        
            fi=fi+1
            
        it=it+1
        print(it)
        
               
    strf = 'iteration, accuracy, auc rocs, auc recall, average precision, # not, # growing \n'
    pi=0
    
    while pi<maxi:    
        score0 = accurs[pi]
        score1 = rocs[pi]
        score2 = recalls[pi]
        score3 = precis[pi]
        
        strc = '%d %f %f %f %f %f %f \n' % (pi, score0, score1, score2, score3, num_0, num_1)
        strf = strf+strc
        pi=pi+1
    
      
    strscocv = 'feature RFE f1 score \n'
    strrank = 'features rank \n'
    strrankcv = 'features ranks cross val \n'
    strselcv = 'feature binary selection \n'
    strcoef = 'feature coefficient \n'
    
    feat_n = len(col_scale)
    
    di=0
    d_uni = list(set(featns))
    # wrtie each row is coeff of feature, column is for different fracture sizes
    for d in d_uni:
        indfs = [i for i, j in enumerate(featns) if j == d] # get all indexes
        
        coef_c = []
        scocv_c = []
        selcv_c = []
        rankcv_c = []
        rank_c = []
        qs_c = []
        wit_c = []
        
        for indf in indfs:
            qs_c.append(qfeats[indf])
            wit_c.append(witers[indf])
            rank_c.append(ranks[indf])
            rankcv_c.append(rankscv[indf])
            scocv_c.append(scorcv[indf])
            selcv_c.append(selcv[indf])
            coef_c.append(coeffs[indf])
        
        strrankcv = strrankcv+col_scale[di]
        strrank = strrank+col_scale[di]
        strscocv = strscocv+str(di+1)
        strselcv = strselcv+col_scale[di]
        strcoef = strcoef+col_scale[di]
        
        for wi in list(set(wit_c)):
            indq = wit_c.index(wi) # only get first index
        
            rankcv_cd = rankcv_c[indq]
            rank_cd = rank_c[indq]
            scocv_cd = scocv_c[indq]
            selcv_cd = selcv_c[indq]
            coef_cd = coef_c[indq]
        
            if selcv_cd==True:
                selcv_cd = 1
            else:
                selcv_cd = 0
        
            strrank = strrank+' '+str(rank_cd)
            strrankcv = strrankcv+' '+str(rankcv_cd)
            strscocv = strscocv+' '+str(scocv_cd)
            strselcv = strselcv+' '+str(selcv_cd)
            strcoef = strcoef+' '+str(coef_cd)
            
        strrank = strrank+'\n'
        strrankcv = strrankcv+'\n'
        strscocv = strscocv+'\n'
        strselcv = strselcv+'\n'
        strcoef = strcoef+'\n'
        di=di+1
    
        
    
    print(txt_name)
    print(strf)
    f= open(txt_name, "w")
    f.write(strf)
    f.close()
    
    txt_name = txt_name.replace('score', 'rank')
    print(txt_name)
    print(strrank)
    f= open(txt_name, "w")
    f.write(strrank)
    f.close()
    
    #print(strrankcv)
    txt_name = txt_name.replace('rank', 'rankcv')
    print(txt_name)
    print(strrankcv)
    f= open(txt_name, "w")
    f.write(strrankcv)
    f.close()
    
    txt_name = txt_name.replace('rankcv', 'scorecv')
    print(txt_name)
    print(strscocv)
    f= open(txt_name, "w")
    f.write(strscocv)
    f.close()
    
    txt_name = txt_name.replace('scorecv', 'selectcv')
    print(txt_name)
    print(strselcv)
    f= open(txt_name, "w")
    f.write(strselcv)
    f.close()
    
    txt_name = txt_name.replace('selectcv', 'coef')
    print(txt_name)
    print(strcoef)
    f= open(txt_name, "w")
    f.write(strcoef)
    f.close()