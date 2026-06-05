from pandas import DataFrame
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def embed_song(text, model):
    words = text.split()

    chunks = [
        " ".join(words[i:i+200])
        for i in range(0, len(words), 200)
    ]

    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )
    mean_embedding = np.mean(chunk_embeddings, axis=0)

    return mean_embedding / np.linalg.norm(mean_embedding)

def preprocess(df):
    # drop unused features and tracks w/ missing vals
    df = (
        df.drop(columns=[
            'rank', 'artist_ids', 'album_name', 'album_id',
            'popularity', 'explicit', 'release_date', 'album_type', 'isrc', 'copies',
            'total_artist_followers', 'avg_artist_popularity', 'artist_genres', 'main_genres',
            'duration', 'duration_ms', 'time_signature', 'key', 'liveness'
        ]) # key and liveness misleading
        .dropna()
        .reset_index(drop=True) # correct indices to reflect dropped rows
    )
    tracklist = df[['track_id', 'track_name', 'artist_names']]

    # tokenize lyrics
    df['lyrics'] = (
        df['lyrics']
        .str.lower()
        .str.replace(r'[\r\n]+', ' ', regex=True)
        .str.replace(r'\[.*?\]', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    # convert key to cyclical features
    # df['key_sin'] = np.sin(df['key']/12 * 2*np.pi)
    # df['key_cos'] = np.cos(df['key']/12 * 2*np.pi)
    # df = df.drop(columns=['key'])

    if os.path.exists("lyrics_embeddings.npy"):
        lyrics = np.load("lyrics_embeddings.npy")
    else:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        lyrics = np.vstack([
            embed_song(text, model)
            for text in df["lyrics"]
        ])
        np.save("lyrics_embeddings.npy", lyrics)
    
    df = df.drop(columns=['lyrics', 'track_id', 'track_name', 'artist_names'])
    audio = df.values

    scaler = StandardScaler()
    audio = scaler.fit_transform(audio)

    return (tracklist, audio, df.columns, lyrics, lyrics.dtype.names)

def search(song_name:str, precomputed=None):
    if precomputed is None:
        tracklist, audio_matrix, audio_col, lyrics_matrix, lyrics_col = preprocess(df)
    else:
        tracklist, audio_matrix, audio_col, lyrics_matrix, lyrics_col = precomputed

    querylist = tracklist[tracklist['track_name'].str.contains(song_name, case=False)]
    return querylist
    

def recommend(df:DataFrame, song:str, n=5, precomputed=None, second_song:str=None):
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
        top_n_idx = sorted_all_idx[sorted_all_idx != song_idx][0:n]
    else:
        top_n_idx = sorted_all_idx[(sorted_all_idx != song_idx) & (sorted_all_idx != second_idx)][0:n]
    top_n_recs = tracklist.iloc[top_n_idx][['track_name', 'artist_names']]
    
    # DEBUGGING TWO SONG SEED
    if second_song is not None:
        second_rank = np.where(sorted_all_idx == second_idx)[0][0]
        print(f"\n--- Debug: {second_song} ---")
        print(f"Rank: {second_rank}, Audio Similarity: {audio_sim[second_idx]:.5f}, Lyrics Similarity: {lyrics_sim[second_idx]:.5f}\n")

        top_n_audio = [audio_matrix[idx] for idx in top_n_idx]
        print(f"Audio Data ({song}, {second_song}, Top {n})")
        print(pd.DataFrame(data=np.vstack((audio_matrix[song_idx], audio_matrix[second_idx])), columns=audio_col))
        print(pd.DataFrame(data=top_n_audio, columns=audio_col))
        top_n_lyrics = [lyrics_matrix[idx] for idx in top_n_idx]
        print(f"Lyrics Data ({song}, {second_song}, Top {n})")
        print(pd.DataFrame(data=np.vstack((lyrics_matrix[song_idx], lyrics_matrix[second_idx])), columns=lyrics_col))
        print(pd.DataFrame(data=top_n_lyrics, columns=lyrics_col))

    return top_n_recs

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data', 'top-10k-spotify-songs-2025-07-detailed.csv')
    df = pd.read_csv(file_path)
    #query = input('Enter the song you would like to find similar tracks to: ')
    #count = input('Enter how many similar tracks to recommend: ')
    query, count = '5jzKL4BDMClWqRguW5qZvh', 10
    print(recommend(df, query, int(count), second_song='3avYqdwHKEq8beXbeWCKqJ'))