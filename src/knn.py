import numpy as np
from src import preprocess

def cosSim(song1, song2):
    song1 = song1[2:].astype(np.float32)
    song2 = song2[2:].astype(np.float32)
    return np.dot(song1, song2) / (np.linalg.norm(song1)*np.linalg.norm(song2))

def knn(song, k=-1, n=1):
    '''
    Finds similar songs from training set to the provided song.

    song : input song array
    k : number of nearest neighbors to compare song to
    n : number of similar songs to be returned in an array
    '''
    if isinstance(k, int) and isinstance(n, int):
        train = preprocess.getTrain()
        if k < 1 or k > len(train):
            k = len(train)
        if n < 1 or k > len(train):
            n = k
        train = train[:k]
        
        return sorted(np.array(
            [
                [trainSong[0], trainSong[1], cosSim(song, trainSong)] for trainSong in train
            ]
        ),
        key=lambda x: x[2],
        reverse=True)[:n]
    else:
        raise ValueError("k nearest neighbors and n similar songs must be integers.")