import numpy as np
import preprocess

raw_test = np.array([
    ["Pure Water", "Migos", "Rap", 2019, 202, "Uh (Woo, woo), no Master P (Ayy) Ten bad bitches and they after me (Bad)"],
    ["Falling", "Chase Atlantic", "Pop", 2018, 99, "And you keep on falling, baby, figure it out Just drive slow, straightforward, or I'm walking around"],
    ["Kids", "MGMT", "Rock", 2005, 122, "Control yourself, take only what you need from it A family of trees wanted to be haunted"],
    ["Maria", "Justin Bieber", "Pop", 2012, 111, "Maria, why you wanna do me like that? That ain't my baby (No), that ain't my girl"],
    ["Alison", "Slowdive", "Rock", 1993, 102, "'Alison,' I said, 'We're sinking' There's nothing here but that's okay"],
])

def cosSim(song1, song2):
    print("cosine similarity between ", song1[0], " and ", song2[0], " is: ")
    song1 = song1[2:].astype(np.float32)
    song2 = song2[2:].astype(np.float32)
    print(np.dot(song1, song2) / (np.linalg.norm(song1)*np.linalg.norm(song2)))
    return np.dot(song1, song2) / (np.linalg.norm(song1)*np.linalg.norm(song2))

def knn(song, k=-1):
    if isinstance(k, int):
        train = preprocess.getTrain()
        if k >= 1 and k <= len(train):
            train = train[:k]
# return song title and artist back
        return sorted(np.array(
            [cosSim(song, trainSong) for trainSong in train]
        ))[-1]
    else:
        raise ValueError("Input must be an integer.")
    
test = preprocess.preprocess(raw_test)
print("cosine similarity:", knn(test[0], 2))