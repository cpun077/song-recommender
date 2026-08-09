import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

script_dir = os.path.dirname(os.path.abspath(__file__))

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

    embedding_path = os.path.join(script_dir, 'data', "lyrics_embeddings.npy")
    if os.path.exists(embedding_path):
        lyrics = np.load(embedding_path)
    else:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        lyrics = np.vstack([
            embed_song(text, model)
            for text in df["lyrics"]
        ])
        np.save(embedding_path, lyrics)

    df = df.drop(columns=['lyrics', 'track_id', 'track_name', 'artist_names'])
    audio = df.values

    scaler = StandardScaler()
    audio = scaler.fit_transform(audio)

    return (tracklist, audio, df.columns, lyrics, lyrics.dtype.names)

def search(song_name:str, precomputed=None, df=None):
    if precomputed is None:
        tracklist, audio_matrix, audio_col, lyrics_matrix, lyrics_col = preprocess(df)
    else:
        tracklist, audio_matrix, audio_col, lyrics_matrix, lyrics_col = precomputed

    querylist = tracklist[tracklist['track_name'].str.contains(song_name, case=False)]
    return querylist
