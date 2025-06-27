import numpy as np

# Title Artist Genre Year BPM Lyrics
train = np.array([
    ["Hot N Cold", "Katy Perry", "Pop", 2008, 132, "'Cause you're hot, then you're cold You're yes, then you're no"],
    ["née-nah", "21 Savage", "Rap", 2024, 165, "When that chopper sing, you really think that they gon' miss you? I spent a half a million dollars on dismissals"],
    ["Come a Little Closer", "Cage The Elephant", "Alternative", 2013, 148, "Come a little closer, then you'll see Come on, come on, come on"],
    ["Jerry Sprunger", "T-Pain", "R&B", 2019, 100, "I'm sprung, dawg she got me Got me doing things I'll never do"],
    ["Clarity", "Zedd", "EDM", 2012, 128, "If our love is tragedy Why are you my remedy? If our love's insanity Why are you my clarity?"],
])

def onehot(arr):
    unique = sorted(set(arr))
    return np.array([[1 if a == u else 0 for u in unique] for a in arr])

def norm(arr):
    arr = arr.astype(np.float32)
    return (arr - np.mean(arr)) / np.std(arr)

print(onehot(train[:,2]))
print(norm(train[:,3]))
print(norm(train[:,4]))

# bag o words reference
# all_words = set(" ".join(lyrics).lower().replace("?", "").replace(",", "").replace("'", "").split())
# vocab = sorted(all_words)
# vocab_dict = {word: i for i, word in enumerate(vocab)}

# def vectorize(text):
#     vec = np.zeros(len(vocab))
#     for word in text.lower().replace("?", "").replace(",", "").replace("'", "").split():
#         if word in vocab_dict:
#             vec[vocab_dict[word]] += 1
#     return vec

# lyrics_vectors = np.array([vectorize(line) for line in lyrics])