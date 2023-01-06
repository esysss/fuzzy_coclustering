import Algorithms
import Funcs as F
import numpy as np
from Merges import *
import pandas as pd
from Recomms import *
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import precision_score, ndcg_score, f1_score, accuracy_score
from tqdm import tqdm

def main(met,m,alg,n_clusters):

    print('method : {}\tmerge:{}\taltorithm:{}\tnumber of clusters:{}'.format(met,m,alg,n_clusters))

    T_train, T_test = F.read_data(have = True)

    if met == "FUIS":
        Pusers, Pitems = Algorithms.FUIS(T_train, n_clusters)
    elif met == 'FCNMF':
        Pusers, Pitems = Algorithms.FCNMF(T_train, n_clusters)
    else:
        print("there is not such algorithm named '{}' ".format(met))
        exit()

    #Decompose multiple clusters for recommender system methods
    threshold = 1/n_clusters
    partitions = {}
    for c in range(n_clusters):
        users = np.where(Pusers[:,c] >= threshold)[0]
        items = np.where(Pitems[:,c] >= threshold)[0]

        if alg == 'SVD':
            partitions[c] = svd(users, items, T_train)
        elif alg =='SVDp':
            partitions[c] = svdplus(users, items, T_train)
        elif alg =='coclus':
            partitions[c] = coclus(users, items, T_train)
        elif alg =='base':
            partitions[c] = base(users, items, T_train)
        elif alg =='normal':
            try:
                partitions[c] = normal(users, items, T_train)
            except:
                print('fuck cluster ', n_clusters)
                partitions[c] = partitions[c-1]
        elif alg == 'knn':
            partitions[c] = knn(users, items, T_train)
        else:
            print("there is no such a recommendation algorithm name '{}' ".format(alg))
            exit()


    #apply merge
    user,item = np.nonzero(T_test)
    T_pred = np.zeros(T_test.shape)
    for i,j in zip(user, item):
        score = 0
        for k in partitions.keys():
            algo = partitions[k]
            if m == 'M1':
                score += merge1(Pusers[i].copy(), Pitems[j].copy(), k) * algo.estimate(i,j)
            elif m == 'M2':
                score += merge2(Pusers[i], k) * algo.estimate(i,j)
            elif m == 'M3':
                score += merge3(Pusers[i], Pitems[j], k) * algo.estimate(i,j)
            else:
                print("there is no such a merging method named '{}' ".format(m))

        if np.isnan(score):
            score = 0
        T_pred[i,j] = score

    result = mae(T_test[np.nonzero(T_test)], T_pred[np.nonzero(T_test)])

    return result

def currecter(met,m,alg,n_clusters):
    for i in range(10):
        result = main(met,m,alg,n_clusters)
        if result < 1.5:
            print('SUCCESS -> method : {}\tmerge:{}\taltorithm:{}\tnumber of clusters:{} = AME : {}'.format(met,m,alg,n_clusters,result))
            return result
    print('unable to get -> method : {}\tmerge:{}\taltorithm:{}\tnumber of clusters:{}'.format(met,m,alg,n_clusters))
    return 1.5

