import numpy as np 
import os 
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt 
embeding_dir = "../embedings"

y_list = []
emb_list = None

# loading data 
for file in os.listdir(embeding_dir):
    path = f"{embeding_dir}/{file}"

    y_list.append(file.split(sep=".")[0])
    
    data = np.load(path)
    emb = data['embedding']
    if emb_list is None:
        emb_list = emb.mean(axis=0)
    else:
        emb_list = np.vstack((emb_list, emb.mean(axis=0)))
    print(path, emb.shape)
    del emb
    del data
print(emb_list)

pca = PCA(n_components=2)
x=pca.fit_transform(emb_list)
print(plt.scatter(x[:0],x[:1]))
plt.savefig("scater.png")
    
#dbscan_cluster_model = DBSCAN(eps=0.9, min_samples=3).fit(emb_list)
#print(dbscan_cluster_model.labels_)


