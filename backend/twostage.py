import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess

def recommend(df:pd.DataFrame, song:str, n=50, k=5, precomputed=None, second_song:str=None):
    """ 
    two-stage recsys: retrieval and reranking \\
    retrieval - union of top n from audio and top n from lyrics \\
    reranking -  top k songs by weighted similarity scoring
    """

    if precomputed is None:
        tracklist, audio_matrix, audio_col, lyrics_matrix, lyrics_col = preprocess(df)
    else:
        tracklist, audio_matrix, audio_col, lyrics_matrix, lyrics_col = precomputed

    try:
        song_idx = tracklist[tracklist['track_id'] == song].index[0]
    except IndexError:
        return "Song not found in database."

    if second_song is None:
        song_audio, song_lyrics = audio_matrix[song_idx], lyrics_matrix[song_idx]
    else:
        second_idx = tracklist[tracklist['track_id'] == second_song].index[0]
        song_audio, song_lyrics = (audio_matrix[song_idx] + audio_matrix[second_idx]) / 2, (lyrics_matrix[song_idx] + lyrics_matrix[second_idx]) / 2

    audio_sim = cosine_similarity(audio_matrix, song_audio.reshape(1,-1)).flatten()
    lyrics_sim = cosine_similarity(lyrics_matrix, song_lyrics.reshape(1,-1)).flatten()

    # (stage 1) retrieval - union of top n from audio and top n from lyrics
    top_audio_idx = np.argsort(-audio_sim)[:n]
    top_lyrics_idx = np.argsort(-lyrics_sim)[:n]
    candidates = np.union1d(top_audio_idx, top_lyrics_idx)

    # exclude seed song(s)
    candidates = candidates[candidates != song_idx]
    if second_song is not None:
        candidates = candidates[candidates != second_idx]

    # (stage 2) reranking - weighted similarity scoring
    candidate_scores = 0.8 * audio_sim[candidates] + 0.2 * lyrics_sim[candidates]
    top_k = candidates[np.argsort(-candidate_scores)[:k]]
    top_k_recs = tracklist.iloc[top_k][['track_name', 'artist_names']]

    return top_k_recs

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data', 'top-10k-spotify-songs-2025-07-detailed.csv')
    df = pd.read_csv(file_path)
    print(recommend(
        df, 
        song='5jzKL4BDMClWqRguW5qZvh',
        n=100, 
        k=10, 
        second_song='3avYqdwHKEq8beXbeWCKqJ'
    ))