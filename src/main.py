import numpy as np 
import os 

embeding_dir = "../embedings"

y_list = []
emb_list = []

for file in os.listdir(embeding_dir):
    path = f"{embeding_dir}/{file}"

    y_list.append(file.split(sep=".")[0])
    
    data = np.load(path)
    emb = data['embedding']
    emb_list.append(emb[-1])
    print(path, emb.shape)
    del emb
    del data
    
print(y_list)
print(emb_list)