from embed import * 
from cluster import *
import matplotlib.pyplot as plt
from scipy.io import wavfile

import os 

import numpy as np

import openl3
import soundfile as sf


DIMENSIONS = 32
RESAMPLE_RATE = 160000

data_dir = "../data"

genre_dirs = os.listdir(data_dir)

count = 0
ds = {} #
 # Output layer for a 32-dimensional fixed-length vector
y = []
audio = []
sr = []
max_len =0 


#model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",
#                                               embedding_size=512)
for genre in genre_dirs:
    wav_files = os.listdir(f"{data_dir}/{genre}")
    for file in wav_files:
        print(f"{data_dir}/{genre}/{file}")
    
        
        data, sr= sf.read(f"{data_dir}/{genre}/{file}")
        emb, ts = openl3.get_audio_embedding(data, sr)# model=model)
        plt.plot(emb[0])
        plt.plot(emb[-1])
        plt.savefig("emb.png")
        
        print(emb.shape)
        print(data, sr)
        exit(1)
   
        count += 1


       # data = data.astype(float)
        
        #audio.append(data)
        #sr.append(samplerate)
        #y.append(file.split(sep=".")[0])

        #print(model.predict(data).shape)
        

#emb_list, ts_list = openl3.get_audio_embedding(audio, sr, batch_size=32)
#print(emb_list)
#print(ts_list)


