# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 23:54:24 2018

@author: Yiming Wen
"""

from function import Load_Data
from function import Load_Names
from numpy import *
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import time


# 计算F-score并按照矩阵输出
# Calculate Fisher score and get the output by matrix
def f_score_core(n_p, n_n, xb, xbp, xbn, xkp, xkn):
    '''
    :param n_p: number of positive features
    :param n_n: number of negative features
    :param xb: list of the average of each feature of the whole instances
    :param xbp: list of the average of each feature of the positive instances
    :param xbn: list of the average of each feature of the negative instances
    :param xkp: list of each feature which is a list of each positive instances
    :param xkn: list of each feature which is a list of each negative instances
    :return: the F-score of the feature
    '''

    def sigma_p (i, n_p, xbp, xkp):
        n_p = n_p.astype('int')
        return sum([(xkp[i][k] - xbp[i])**2 for k in range(n_p)])

    def sigma_n(i, n_n, xbn, xkn):
        n_n = n_n.astype('int')
        return sum([(xkn[i][k] - xbn[i])**2 for k in range(n_n)])

    n_feature = len(xb)
    F_scores = []
    for i in range(n_feature):
        F_score_numerator = (xbp[i]-xb[i])**2 + (xbn[i]-xb[i])**2
        F_score_denominator = (1/float(n_p-1))*(sigma_p(i,n_p,xbp,xkp)) + (1/float(n_n-1))*(sigma_n(i,n_n,xbn,xkn))
        F_scores.append(F_score_numerator/F_score_denominator)
    return F_scores


# 提取数据内容为计算F-score提供参数
# Extract information from featureset and provide parameters for calculating Fisher score
def f_score(feature, classindex):
    '''
    :param feature: a matrix whose rows indicate instances, columns indicate features
    :param classindex: 1 indicates positive and 0 indicates negative
    :return: The F-score matrix
    '''
    n_instance = len(feature)
    n_feature = len(feature[0])
    n_p = sum (classindex)
    n_n = n_instance - n_p
    xkp = []; xkn = []; xbp = []; xbn = []; xb = []
    for i in range(n_feature):
        xkp_i = []; xkn_i = []
        for j in range(n_instance):
            if classindex[j] == 1:
                xkp_i.append(feature[j][i])
            else:
                xkn_i.append(feature[j][i])
        xkp.append(xkp_i)
        xkn.append(xkn_i)
        sum_xkp_i = sum(xkp_i)
        sum_xkn_i = sum(xkn_i)
        xbp.append(sum_xkp_i / float(n_p))
        xbn.append(sum_xkn_i / float(n_n))
        xb.append((sum_xkp_i+sum_xkn_i)/float(n_instance))
    return f_score_core(n_p, n_n, xb, xbp, xbn, xkp, xkn)


def main():
    ## 开始计时
    ## Start counting time
    start = time.clock()


    ## 载入数据
    ## Load dataset
    Dataset, fileName = Load_Data.loadtxt('SMK_CAN_187.txt')
    if fileName == 'Ovarian_Cancer.txt':
        C = 0.055
    elif fileName == 'SMK_CAN_187.txt':
        C = 0.005
    elif fileName == 'madelon.txt':
        C = 0.009
    

    ## 随机抽取三分之二作为训练集，另外三分之一作为测试集
    ## Randomly choose 66% of dataset as training data and others as test data
    train, test = Load_Data.data_selection(Dataset, .66)
    print('\nSize of training set:{0}\n'.format(np.shape(train)[0]))
    print('\nSize of test set:{0}\n'.format(np.shape(test)[0]))
    x_train, y_train = Load_Data.split_data(train)
    x_test, y_test = Load_Data.split_data(test)
    y_train = Load_Names.Namelist(fileName, y_train)
    y_test = Load_Names.Namelist(fileName, y_test)
        
    
#    for i in range(len(y_train)):
#        if y_train[i] == -1:
#            y_train[i] = 0
    
    
    ## 计算F-score
    ## Calculate Fisherscore
    F_score = f_score(x_train, y_train)
    
    
    ## 选取前10个Fisher score
    ## Select the first 10 fisher score
    # Make a list of (Fisher score, Featureset No.) tuples
    F_score_pairs = [(F_score[i], i) for i in range(len(x_train[0]))]
    
    
    # Sort the (Fisher score, Featureset No.) tuples from high to low
    F_score_pairs.sort(key=lambda x: x[0], reverse=True)
    
        
    # 选择K个Fisher score最大的特征列
    # Select K features with larger Fisher scores 
    K_features = 20
    Feature_column = []
    for i in range(K_features):
        Feature_column.append(F_score_pairs[i][1])
        
        
    print(Feature_column)
    
    
    x_train_N = []
    x_test_N = []
    for i in range(K_features):
        if i == 0:
            x_train_N = hstack((x_train_N, x_train[:, Feature_column[i]]))
            x_test_N = hstack((x_test_N, x_test[:, Feature_column[i]]))
        else:
            x_train_N = vstack((x_train_N, x_train[:, Feature_column[i]]))
            x_test_N = vstack((x_test_N, x_test[:, Feature_column[i]]))

    x_train = np.transpose(x_train_N)
    x_test = np.transpose(x_test_N)

    
    ## 训练svm分类器
    ## Train SVM classifier
    classifier = svm.SVC(C = C, kernel='linear', decision_function_shape='ovr')
    #classifier = svm.SVC(C = 0.1, kernel='rbf', gamma=10, decision_function_shape='ovr')
    classifier.fit(x_train, y_train.ravel())
        
    
    ## 计算svc分类器的准确率
    ## Calculate the accuracy of SVM
    # Accuracy
    #print (classifier.score(x_train, y_train))
    y_hat = classifier.predict(x_train)
    print('\nTraining set accuracy: \n{0}\n'.format(accuracy_score(y_hat, y_train, '训练集 / Train data')))
    
    #print(classifier.score(x_test, y_test))
    y_hat = classifier.predict(x_test)
    print('\nTest set accuracy: \n{0}\n'.format(accuracy_score(y_hat, y_test, '测试集 / Test data')))
        
    #    # 决策函数
    #    # Decision function
    #    print('Decision function of training set:\n{0}'.format(classifier.decision_function(x_train)))
    #    print('\nPrediction of training set:\n{0}'.format(classifier.predict(x_train)))
    #    print('Decision function of test set:\n{0}'.format(classifier.decision_function(x_test)))
    #    print('\nPrediction of test set :\n{0}\n\n'.format(classifier.predict(x_test)))
        
        
    end = time.clock()
    print('Running time: %s Seconds\n\n'%(end-start))

    i  
    
if __name__ == "__main__":
    main()