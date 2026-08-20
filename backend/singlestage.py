import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess

def recommend(df:pd.DataFrame, song:str, k=5, precomputed=None, second_song:str=None):
    """ 
    single-stage recsys: top k songs by fusion ranking
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

    audio_rank = np.argsort(np.argsort(-audio_sim))
    lyrics_rank = np.argsort(np.argsort(-lyrics_sim))
    combined_rank = (
        0.8 * audio_rank + 0.2 * lyrics_rank
        #audio_rank
    )
    sorted_all_idx = np.argsort(combined_rank)

    if second_song is None:
        top_n_idx = sorted_all_idx[sorted_all_idx != song_idx][0:k]
    else:
        top_n_idx = sorted_all_idx[(sorted_all_idx != song_idx) & (sorted_all_idx != second_idx)][0:k]
    top_n_recs = tracklist.iloc[top_n_idx][['track_id', 'track_name', 'artist_names']]
    
    # DEBUGGING TWO SONG SEED
    # if second_song is not None:
    #     second_rank = np.where(sorted_all_idx == second_idx)[0][0]
    #     print(f"\n--- Debug: {second_song} ---")
    #     print(f"Rank: {second_rank}, Audio Similarity: {audio_sim[second_idx]:.5f}, Lyrics Similarity: {lyrics_sim[second_idx]:.5f}\n")

    #     top_n_audio = [audio_matrix[idx] for idx in top_n_idx]
    #     print(f"Audio Data ({song}, {second_song}, Top {k})")
    #     print(pd.DataFrame(data=np.vstack((audio_matrix[song_idx], audio_matrix[second_idx])), columns=audio_col))
    #     print(pd.DataFrame(data=top_n_audio, columns=audio_col))
    #     top_n_lyrics = [lyrics_matrix[idx] for idx in top_n_idx]
    #     print(f"Lyrics Data ({song}, {second_song}, Top {k})")
    #     print(pd.DataFrame(data=np.vstack((lyrics_matrix[song_idx], lyrics_matrix[second_idx])), columns=lyrics_col))
    #     print(pd.DataFrame(data=top_n_lyrics, columns=lyrics_col))

    return top_n_recs

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data', 'top-10k-spotify-songs-2025-07-detailed.csv')
    df = pd.read_csv(file_path)
    print(recommend(
        df, 
        song='5jzKL4BDMClWqRguW5qZvh',
        k=10, 
        second_song='3avYqdwHKEq8beXbeWCKqJ'
    ))