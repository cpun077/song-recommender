from src import preprocess
from src import knn
import numpy as np

# Title Artist Genre Year BPM Lyrics
# add csv read that removes headers
raw_train = np.array([
    ["Hot N Cold", "Katy Perry", "Pop", 2008, 132, "'Cause you're hot, then you're cold You're yes, then you're no"],
    ["née-nah", "21 Savage", "Rap", 2024, 165, "When that chopper sing, you really think that they gon' miss you? I spent a half a million dollars on dismissals"],
    ["Come a Little Closer", "Cage The Elephant", "Alternative", 2013, 148, "Come a little closer, then you'll see Come on, come on, come on"],
    ["Jerry Sprunger", "T-Pain", "R&B", 2019, 100, "I'm sprung, dawg she got me Got me doing things I'll never do"],
    ["Clarity", "Zedd", "EDM", 2012, 128, "If our love is tragedy Why are you my remedy? If our love's insanity Why are you my clarity?"],
])

def main():
    data = preprocess.preprocess(raw_train) # numericalize
    print(data)
    preprocess.newTrain(data) # create new train csv
    print(preprocess.getTrain())

if __name__ == "__main__":
    main()