import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def calc_norms(liked_song, all_songs):
    norms = []
    
    for s in all_songs:

        n = np.linalg.norm(s-liked_song)
        norms.append(n)

    return norms

def calc_cosine_sim(liked_song, all_songs):
    norms = []
    
    for s in all_songs:

        n = cosine_similarity([s], [liked_song])
        norms.append(n)

    return norms

def min_index_nonzero(arr):
    m = -1
    for i in range(len(arr)):
        if arr[i] > 0 and (m == -1 or arr[i] < arr[m]):
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
    

def recomend_sort(liked_indexes, all_songs):
    
    
    sorted_dists = []
    for liked in liked_indexes:
        norms = calc_norms(liked, all_songs)
        for i in range(len(all_songs)):
            norms[i] = (i, norms[i]) 
        
        norms = sorted(norms, key=lambda x: x[1])
        sorted_dists.append(norms)
    
    canidate_map = {}
    for i in range(len(all_songs)):
        for dists_from_liked in sorted_dists:
            canidate = dists_from_liked[i][0]
            canidate_count = canidate_map.get(canidate,0)
            canidate_count += 1
            if canidate_count == len(liked_indexes):
                return canidate
            canidate_map[canidate] = canidate_count

def max_index(arr):
    m = 0 
    for  i in range(len(arr)):
        if arr[i]> arr[m]:
            m = i 
    return m

def recomend_cosine(liked_indexs, all_songs):
    sum_dists = None
    
    
    for i in liked_indexs:
        liked_song = all_songs[i]
        n = calc_cosine_sim(liked_song, all_songs) 
        
        if sum_dists is None:
            sum_dists = n
        else:
            for j in range(len(n)):
                sum_dists[j]+= n[j]
        
    return min_index_nonzero(sum_dists)
    
    
        
    




