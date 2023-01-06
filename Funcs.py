from fcmeans import FCM
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import precision_score, ndcg_score, f1_score, accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_absolute_error as mae
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph

def read_data(have = True, information = False):

    if have:
        pickleIN = open("Dataset/Movie_rates.p", "rb")  # says read it to bite
        T_train, T_test = pickle.load(pickleIN)
        pickleIN.close()
        return T_train, T_test
    else:
        # read the data for both training and testing
        df = pd.read_csv('Dataset/u.data', sep='\t', names=["user id", "item id", "rating", "timestamp"])

        if information:
            print("the whole data has {} users and {} items with {} rates".
                  format(len(df['user id'].unique()), len(df['item id'].unique()), len(df)))

        # read the training data
        df_train = pd.read_csv('Dataset/ua.base', sep='\t', names=["user id", "item id", "rating", "timestamp"])

        if information:
            print("number of training data samples: ", len(df_train))
            print('number of users: ', len(df_train['user id'].unique()))
            print("number of items: ", len(df_train['item id'].unique()))

        # read the testing data
        df_test = pd.read_csv('Dataset/ua.test', sep='\t', names=["user id", "item id", "rating", "timestamp"])

        if information:
            print("number of training data samples: ", len(df_test))
            print('number of users: ', len(df_test['user id'].unique()))
            print("number of items: ", len(df_test['item id'].unique()))

        # constructing the matrix T for training data
        T_train = np.zeros((len(df['user id'].unique()), len(df['item id'].unique())),dtype=int)

        if information:
            print("The matrix T(train) is getting made\nthe shape of T is ", T_train.shape)

        for i in tqdm(range(len(df_train))):
            T_train[df_train['user id'].iloc[i] - 1, df_train['item id'].iloc[i] - 1] = df_train['rating'].iloc[i]

        # constructing the matrix T for testing data
        T_test = np.zeros((len(df['user id'].unique()), len(df['item id'].unique())),dtype=int)

        if information:
            print("The matrix T(test) is getting made\nthe shape of T is ", T_test.shape)

        for i in tqdm(range(len(df_test))):
            T_test[df_test['user id'].iloc[i] - 1, df_test['item id'].iloc[i] - 1] = df_test['rating'].iloc[i]

        if information:
            print("matrix T(train) is ", T_train.shape)


        T_train[674,1652] = T_test[674,1652]
        T_train[404,1581] = T_test[404,1581]

        theFile = open("Dataset/Movie_rates.p", "wb")  # it says to write in bite
        pickle.dump((T_train, T_test), theFile)
        theFile.close()

        return T_train, T_test


def dist(T_train, metric):

    #metric = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']

    user_similarity = pairwise_distances(T_train, metric=metric)
    item_similarity = pairwise_distances(T_train.T, metric=metric)

    return user_similarity, item_similarity

def cmeans(X, n_clusters):
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(X)

    return fcm.u


#evaluations
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def printer(y_actual, y_predict):
    TP, FP, TN, FN = perf_measure(y_actual, y_predict)
    f1 = f1_score(y_actual, y_predict)
    recall = recall_score(y_actual, y_predict)
    ACC = (TP + TN) / (TP + FP + FN + TN)

    if TP != 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 0

    if TN != 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 0

    if TN !=0:
        TNR = TN / (TN + FP)
    else:
        TNR = 0

    return ACC, PPV, NPV, recall, TNR, f1, TP, FP, TN, FN


def T_bar(T_train):
    g_num = np.sum(T_train, axis=-1)
    q_num = np.sum(T_train, axis=-2)

    g_denum = np.nonzero(T_train)[0]
    q_denum = np.nonzero(T_train)[1]

    g = np.nan_to_num(g_num/np.bincount(g_denum))

    q = np.nan_to_num(q_num/np.bincount(q_denum))

    add1,add2 = np.nonzero(T_train == 0)

    for i,j in zip(add1,add2):
        T_train[i,j] = (g[i] + q[j]) / 2

    return T_train

def knn(T_train):

    WU = kneighbors_graph(T_train, 25, mode='connectivity', include_self=True)
    WI = kneighbors_graph(T_train.T, 25, mode='connectivity', include_self=True)

    return WU.toarray(), WI.toarray()