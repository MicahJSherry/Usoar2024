import numpy as np 
import os 
from sklearn.cluster import DBSCAN
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
        emb_list = emb.mean()
    else:
        emb_list = np.vstack((emb_list, emb.mean()))
    print(path, emb.shape)
    del emb
    del data
print(emb_list)

x = emb.sum(axis=1)
plt.scatter(x,x)
plt.savefig("scater.png")
    
#dbscan_cluster_model = DBSCAN(eps=0.9, min_samples=3).fit(emb_list)
#print(dbscan_cluster_model.labels_)


