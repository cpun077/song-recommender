import numpy as np
import pandas as pd
import re
import os

headers = ['title','artist','genre','year','bpm','lyrics']

def vectorNorm(vector): # compress matrix of vectors with Euclidian norm ("magnitude") for single value
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
    frame.columns = headers
    frame.to_csv(outPath, index=False)

def updateTrain(newData, outPath="./data/train.csv"):
    if isinstance(newData, str) and os.path.isfile(newData):
        frame = pd.read_csv(newData)
        newDataHeaders = np.array(frame.columns)
        if not all(newDataHeaders == headers): # doesn't check if header is a valid song
            frame.to_csv(outPath, mode='a', header=True, index=False)
            return
    elif isinstance(newData, (np.ndarray, list)):
        frame = pd.DataFrame(newData)
    else:
        raise ValueError("Input must be either a valid file path (str) or an array-like object.")
    
    frame.to_csv(outPath, mode='a', header=False, index=False)

def getTrain(path="./data/train.csv"):
    return pd.read_csv(path).values

def preprocess(train):
    name = train[:,0].reshape(-1,1)
    artist = train[:,1].reshape(-1,1)
    genre = onehot(train[:,2])
    year = norm(train[:,3]).reshape(-1,1)
    bpm = norm(train[:,4]).reshape(-1,1)
    lyrics = bow(train[:,5])

    return np.hstack([name, artist, genre, year, bpm, lyrics])