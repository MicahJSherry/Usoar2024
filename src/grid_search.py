import numpy as np 
import os 
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt 



# loading data 
embeding_dir = "../embedings"
figs = "../figs"
emb_list = None
i = 0

for file in os.listdir(embeding_dir):
    path = f"{embeding_dir}/{file}"
    tokens=file.split(sep=".")
    print(i,tokens)
    
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


def dbscan_parameter_search(X, eps_range, min_samples_range):
  
  best_params = {'eps': None, 'min_samples': None}
  best_score = -1

  for eps in eps_range:
    for min_samples in min_samples_range:
      dbscan = DBSCAN(eps=eps, min_samples=min_samples)
      labels = dbscan.fit_predict(X)
      
      # Skip if all points are noise
      if len(set(labels)) == 1 and -1 in labels:
        continue

      score = silhouette_score(X, labels)
      print(eps, min_samples, score, len(set(labels)))
      if score > best_score:
        best_params['eps'] = eps
        best_params['min_samples'] = min_samples
        best_score = score

  return best_params, best_score

# Example usage:


eps_range = np.linspace(0.1, 10, 10)
min_samples_range = range(5, 20)


p , s =dbscan_parameter_search(emb_list, eps_range, min_samples_range)

dbscan  = DBSCAN(eps=p["eps"], min_samples=p["min_samples"]).fit(emb_list)
y_dbscan = dbscan.labels_

plt.scatter(x[:,0],x[:,1], c=y_dbscan, cmap='viridis')
plt.title("music grouped by DBSCAN clustering")
plt.savefig(f"{figs}/scater_DBSCAN.png")
plt.clf()