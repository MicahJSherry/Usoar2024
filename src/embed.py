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

emb_list = []

model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=512)
for genre in genre_dirs:
    wav_files = os.listdir(f"{data_dir}/{genre}")
    for file in wav_files:
        #print(f"{data_dir}/{genre}/{file}")

        try:
            filepath = f"{data_dir}/{genre}/{file}"
            openl3.process_audio_file(filepath, output_dir='../embedings', model=model)
        except:
            count += 1

