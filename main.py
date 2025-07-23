from src import preprocess
from src import knn
import numpy as np

# Title Artist Genre Year BPM Lyrics
raw_train = np.array([
    ["Hot N Cold", "Katy Perry", "Pop", 2008, 132, "'Cause you're hot, then you're cold You're yes, then you're no"],
    ["née-nah", "21 Savage", "Rap", 2024, 165, "When that chopper sing, you really think that they gon' miss you? I spent a half a million dollars on dismissals"],
    ["Come a Little Closer", "Cage The Elephant", "Alternative", 2013, 148, "Come a little closer, then you'll see Come on, come on, come on"],
    ["Jerry Sprunger", "T-Pain", "R&B", 2019, 100, "I'm sprung, dawg she got me Got me doing things I'll never do"],
    ["Clarity", "Zedd", "EDM", 2012, 128, "If our love is tragedy Why are you my remedy? If our love's insanity Why are you my clarity?"],
])
raw_test = np.array([
    ["Pure Water", "Migos", "Rap", 2019, 202, "Uh (Woo, woo), no Master P (Ayy) Ten bad bitches and they after me (Bad)"],
    ["Falling", "Chase Atlantic", "Pop", 2018, 99, "And you keep on falling, baby, figure it out Just drive slow, straightforward, or I'm walking around"],
    ["Kids", "MGMT", "Rock", 2005, 122, "Control yourself, take only what you need from it A family of trees wanted to be haunted"],
    ["Maria", "Justin Bieber", "Pop", 2012, 111, "Maria, why you wanna do me like that? That ain't my baby (No), that ain't my girl"],
    ["Alison", "Slowdive", "Rock", 1993, 102, "'Alison,' I said, 'We're sinking' There's nothing here but that's okay"],
])

def main():
    # uploading new train
    # data = preprocess.preprocess(raw_train)
    # preprocess.newTrain(data)

    test = preprocess.preprocess(raw_test)
    results = np.array(
        [[testsong[0], knn.knn(testsong)[0][0]] for testsong in test]
    )
    for x in results:
        print('If you like', x[0], 'then listen to', x[1])

if __name__ == "__main__":
    main()