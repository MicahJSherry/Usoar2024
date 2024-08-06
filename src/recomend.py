import numpy as np

def calc_norms(liked_song, all_songs):
    norms = []
    
    for s in all_songs:

        n = np.linalg.norm(s-liked_song)
        norms.append(n)

    return norms

def min_index_nonzero(arr):
    m = 0
    for i in range(len(arr)):
        if arr[i] > 0 and arr[i] < arr[m]:
            m = i 
    return m 



def recomend_sum(liked_indexs, all_songs):
    sum_dists = None
    
    
    for i in liked_indexs:
        liked_song = all_songs[i]
        n = calc_norms(liked_song, all_songs) 
        
        if sum_dists is None:
            sum_dists = n
        else:
            for j in range(len(n)):
                if n[j] == 0 or sum_dists[j] == 0:
                    sum_dists[j] = 0
                else:
                    sum_dists[j]+= n[j]
    return min_index_nonzero(sum_dists)
    
    
    
    
    




