import numpy as np 
import os 
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt 
from recomend import *

def get_track_str(i,genres,tracks):
    return f"{genres[i]}.{tracks[i]}"

embeding_dir = "../embedings"

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

pca = PCA(n_components=2)
x=pca.fit_transform(emb_list)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_list)
    
#print(y_encoded)

kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(emb_list)
y_kmeans = kmeans.labels_

plt.scatter(x[:,0],x[:,1],c=y_encoded, cmap='viridis')
plt.title("music grouped by Genre")
plt.savefig("scater_true.png")
plt.clf()

plt.scatter(x[:,0],x[:,1], c=y_kmeans, cmap='viridis')
plt.title("music grouped by K-Means clustering")
plt.savefig("scater_kmeans.png")

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