import numpy as np
import pandas as pd
import re

# Title Artist Genre Year BPM Lyrics
labels = ['title','artist','genre','year','bpm','lyrics']
raw_train = np.array([
    ["Hot N Cold", "Katy Perry", "Pop", 2008, 132, "'Cause you're hot, then you're cold You're yes, then you're no"],
    ["née-nah", "21 Savage", "Rap", 2024, 165, "When that chopper sing, you really think that they gon' miss you? I spent a half a million dollars on dismissals"],
    ["Come a Little Closer", "Cage The Elephant", "Alternative", 2013, 148, "Come a little closer, then you'll see Come on, come on, come on"],
    ["Jerry Sprunger", "T-Pain", "R&B", 2019, 100, "I'm sprung, dawg she got me Got me doing things I'll never do"],
    ["Clarity", "Zedd", "EDM", 2012, 128, "If our love is tragedy Why are you my remedy? If our love's insanity Why are you my clarity?"],
])

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

def export(data, labels, filename="./data/train.csv"):
    frame = pd.DataFrame(data)
    frame.columns = labels
    frame.to_csv(filename)

def main():
    name = raw_train[:,0].reshape(-1,1)
    artist = raw_train[:,1].reshape(-1,1)
    genre = onehot(raw_train[:,2])
    year = norm(raw_train[:,3]).reshape(-1,1)
    bpm = norm(raw_train[:,4]).reshape(-1,1)
    lyrics = bow(raw_train[:,5])

    data = np.hstack([name, artist, genre, year, bpm, lyrics])

    print(data)
    export(data, labels)

if __name__ == "__main__":
    main()