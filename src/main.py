import numpy as np 
import os 
from sklearn.cluster import HDBSCAN, DBSCAN, KMeans
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE 

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt 
from sklearn.metrics import  silhouette_score

from recomend import *
    


embeding_dir = "../embedings"
figs = "../figs"
y_list = []
track_nums= []
emb_list = None
i = 0

# loading data 
for file in os.listdir(embeding_dir):
    path = f"{embeding_dir}/{file}"
    tokens=file.split(sep=".")
    print(i,tokens)
    y_list.append(tokens[0])
    track_nums.append(tokens[1])
    
    data = np.load(path)
    emb = data['embedding']
    if emb_list is None:
        emb_list = emb.mean(axis=0)
    else:
        emb_list = np.vstack((emb_list, emb.mean(axis=0)))
    #print(path, emb.shape)
    del emb
    del data
    i += 1
print(emb_list.shape)

pca = TSNE(n_components=2)
x=pca.fit_transform(emb_list)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_list)

ss_genre = silhouette_score(emb_list, y_encoded)    
#print(y_encoded)

kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(emb_list)
y_kmeans = kmeans.labels_
ss_kmeans = silhouette_score(emb_list, y_kmeans)    

hdbscan = HDBSCAN().fit(emb_list)
y_hdbscan = hdbscan.labels_
ss_hdbscan = silhouette_score(emb_list, y_hdbscan) 

dbscan= DBSCAN().fit(emb_list)
y_dbscan = dbscan.labels_
ss_dbscan = silhouette_score(emb_list, y_dbscan) 



plt.scatter(x[:,0],x[:,1],c=y_encoded, cmap='viridis')
plt.title("music grouped by Genre \tsilhouette_score:{ss_genre}")
plt.savefig(f"{figs}/scater_true.png")
plt.clf()

plt.scatter(x[:,0],x[:,1], c=y_kmeans, cmap='viridis')
plt.title("music grouped by K-Means clustering \tsilhouette_score:{ss_kmeans}")
plt.savefig(f"{figs}/scater_kmeans.png")
plt.clf()


plt.scatter(x[:,0],x[:,1], c=y_hdbscan, cmap='viridis')
plt.title(f"music grouped by HDBSCAN clustering \tsilhouette_score:{ss_hdbscan} ")
plt.savefig(f"{figs}/scater_HDBSCAN.png")
plt.clf()


plt.scatter(x[:,0],x[:,1], c=y_dbscan, cmap='viridis')
plt.title("music grouped by DBSCAN clustering \tsilhouette_score:{ss_dbscan}")
plt.savefig(f"{figs}/scater_DBSCAN.png")
plt.clf()


def recomend(arr):
    print("given this set of liked songs:")
    for i in arr:
        print(f"{i}: {y_list[i]}.{track_nums[i]}")
    
    r1 = recomend_sum(arr,emb_list)
    r2 = recomend_sort(arr,emb_list)
    r3 = recomend_cosine(arr,emb_list)
    print(f"recomended (by sum algorithm)  song is {r1}: {y_list[r1]}.{track_nums[r1]}")
    print(f"recomended (by sort algorithm) song is {r2}: {y_list[r2]}.{track_nums[r2]}")
    print(f"recomended (by sort algorithm) song is {r3}: {y_list[r3]}.{track_nums[r3]}")
    print()

arrs = [
    [988, 989, 990],
    [388, 259],
    [801,782,802],
    [780,781]
    ]
for arr in arrs:
    recomend(arr)
