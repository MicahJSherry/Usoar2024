import numpy as np 
import os 
from sklearn.cluster import DBSCAN
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
        emb_list = emb[-1]
    else:
        emb_list = np.append(emb_list, emb[-1])
    print(path, emb.shape)
    del emb
    del data
print(emb_list.shape)

