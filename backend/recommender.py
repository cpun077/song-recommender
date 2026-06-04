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
    

def recommend(df:DataFrame, song:str, n=5, precomputed=None):
    if precomputed is None:
        tracklist, audio_matrix, audio_col, lyrics_matrix, lyrics_col = preprocess(df)
    else:
        tracklist, audio_matrix, audio_col, lyrics_matrix, lyrics_col = precomputed

    weights = np.array([
        1.2,  # danceability
        1.2,  # energy
        1.0,  # loudness
        0.1,  # mode
        0.3,  # speechiness
        0.8,  # acousticness
        0.5,  # instrumentalness
        1.3,  # valence
        1.0   # tempo
    ])
    audio_matrix = audio_matrix * weights

    try:
        song_idx = tracklist[tracklist['track_id'] == song].index[0]
    except IndexError:
        return "Song not found in database."

    song_audio, song_lyrics = audio_matrix[song_idx], lyrics_matrix[song_idx]

    audio_sim = cosine_similarity(audio_matrix, song_audio.reshape(1,-1)).flatten()
    lyrics_sim = cosine_similarity(lyrics_matrix, song_lyrics.reshape(1,-1)).flatten()

    audio_rank = np.argsort(np.argsort(-audio_sim))
    lyrics_rank = np.argsort(np.argsort(-lyrics_sim))

    combined_rank = (
        0.8 * audio_rank + 0.2 * lyrics_rank
        #audio_rank
    )
    sorted_all_idx = np.argsort(combined_rank)
    top_n_idx = sorted_all_idx[1:n+1]
    top_n_recs = tracklist.iloc[top_n_idx][['track_name', 'artist_names']]
    
    # DEBUGGING TEENAGE DREAM / LAST FRIDAY NIGHT (TGIF)
    target_song = '3avYqdwHKEq8beXbeWCKqJ'
    target_idx = tracklist[tracklist['track_id'] == target_song].index[0]
    target_rank = np.where(sorted_all_idx == target_idx)[0][0]
    print(f"\n--- Debug: {target_song} ---")
    print(f"Rank: {target_rank}, Audio Similarity: {audio_sim[target_idx]:.5f}, Lyrics Similarity: {lyrics_sim[target_idx]:.5f}\n")

    top_n_audio = [audio_matrix[idx] for idx in top_n_idx]
    print(f"Audio Data ({song}, {target_song}, Top {n})")
    print(pd.DataFrame(data=np.vstack((audio_matrix[song_idx], audio_matrix[target_idx])), columns=audio_col))
    print(pd.DataFrame(data=top_n_audio, columns=audio_col))
    top_n_lyrics = [lyrics_matrix[idx] for idx in top_n_idx]
    print(f"Lyrics Data ({song}, {target_song}, Top {n})")
    print(pd.DataFrame(data=np.vstack((lyrics_matrix[song_idx], lyrics_matrix[target_idx])), columns=lyrics_col))
    print(pd.DataFrame(data=top_n_lyrics, columns=lyrics_col))

    return top_n_recs

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data', 'top-10k-spotify-songs-2025-07-detailed.csv')
    df = pd.read_csv(file_path)
    #query = input('Enter the song you would like to find similar tracks to: ')
    #count = input('Enter how many similar tracks to recommend: ')
    query, count = '5jzKL4BDMClWqRguW5qZvh', 10
    print(recommend(df, query, int(count)))
    print(search('Teenage Dream'))