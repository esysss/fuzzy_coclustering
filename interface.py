import warnings
warnings.filterwarnings("ignore")

import inter
import numpy as np

# merges = ['M1', 'M2', 'M3']
merges = ['M3']
methods = ['SVD', 'SVDp', 'base', 'normal']
algorithms = ['FCNMF','FUIS']

counter = 3

for met in algorithms:
    for m in merges:
        for alg in methods:
            counter-=1
            if counter < 0:
                results = []
                for n_clusters in range(5,61,5):
                    result = inter.currecter(met,m,alg,n_clusters)
                    results.append(result)
                    print("#############################")
                np.save('Results/{}_{}_{}.npy'.format(alg,met,m), results)