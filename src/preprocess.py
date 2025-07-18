import numpy as np
import pandas as pd
import re
import os

def vectorNorm(vector): # compress vectors with Euclidian norm for single value
    return np.linalg.norm(vector, axis=1).reshape(-1, 1)

def onehot(arr):
    unique = sorted(set(arr))
    return vectorNorm(np.array(
        [
            [1 if elem == uelem else 0 for uelem in unique] for elem in arr
        ]
    ))

def norm(arr):
    arr = arr.astype(np.float32)
    return (arr - np.mean(arr)) / np.std(arr)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z ]", "", text.lower())
    return text.split()

def freqVector(text, dict):
    vector = np.zeros(len(dict))
    for word in text:
        if word in dict:
            vector[dict[word]] += 1
    return vector

def bow(arr):
    tokenArr = [tokenize(text) for text in arr]
    bank = sorted(set([word for song in tokenArr for word in song]))
    dict = {word: i for i, word in enumerate(bank)}
    return vectorNorm(np.array(
        [freqVector(text, dict) for text in tokenArr]
    ))

def newTrain(inArr, outPath="./data/train.csv"):
    frame = pd.DataFrame(inArr)
    frame.columns = ['title','artist','genre','year','bpm','lyrics']
    frame.to_csv(outPath, index=False)

def updateTrain(newData, outPath="./data/train.csv"):
    if isinstance(newData, str) and os.path.isfile(newData):
        frame = pd.read_csv(newData)
    elif isinstance(newData, (np.ndarray, list)):
        frame = pd.DataFrame(newData)
    else:
        raise ValueError("Input must be either a file path (str) or an array-like object.")
    
    frame.to_csv(outPath, mode='a', header=False, index=False)

def preprocess(train):
    name = train[:,0].reshape(-1,1)
    artist = train[:,1].reshape(-1,1)
    genre = onehot(train[:,2])
    year = norm(train[:,3]).reshape(-1,1)
    bpm = norm(train[:,4]).reshape(-1,1)
    lyrics = bow(train[:,5])

    return np.hstack([name, artist, genre, year, bpm, lyrics])