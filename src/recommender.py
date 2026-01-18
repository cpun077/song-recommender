import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(df):
    # drop unnecessary col and rows w/ missing vals
    df = df.drop(columns=[
        'rank', 'track_name', 'artist_names', 'artist_ids', 'album_name', 'album_id', \
        'popularity', 'explicit', 'release_date', 'album_type', 'isrc', 'copies', \
        'total_artist_followers', 'avg_artist_popularity', 'artist_genres', 'main_genres'\
    ]).dropna()

    # convert duration and tokenize lyrics
    df['duration'] = pd.to_timedelta('00:' + df['duration']).dt.total_seconds().astype(int)
    df['lyrics'] = df['lyrics'] \
    .str.lower() \
    .str.replace(r'[\r\n]+', ' ', regex=True) \
    .str.replace(r'\[.*?\]', '', regex=True) \
    .str.replace(r'[^A-Za-z\d\s]', '', regex=True) \
    .str.replace(r'\s+', ' ', regex=True) \
    .str.strip()
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', max_df=0.8)
    lyrics = vectorizer.fit_transform(df['lyrics']).toarray()
    df = df.drop(columns=['lyrics', 'track_id'])
    audio = df.values

    scaler = MinMaxScaler()
    lyrics_scaled = scaler.fit_transform(lyrics)
    audio_scaled = scaler.fit_transform(audio)
    return (audio_scaled, df.columns, lyrics_scaled, vectorizer.get_feature_names_out().tolist())

def recommend(df, song, n=5):
    try:
        song_idx = df[df['track_name'] == song].index[0]
    except IndexError:
        return "Song not found in database."

    audio_matrix, audio_col, lyrics_matrix, lyrics_col = preprocess(df)
    song_audio, song_lyrics = audio_matrix[song_idx], lyrics_matrix[song_idx]

    audio_similarity = cosine_similarity(audio_matrix, song_audio.reshape(1,-1))
    lyrics_similarity = cosine_similarity(lyrics_matrix, song_lyrics.reshape(1,-1))
    sim_matrix = 0.9 * audio_similarity + 0.1 * lyrics_similarity

    top_n_idx = sim_matrix.flatten().argsort()[::-1][1:n+1] # reverse sort idxs, skip queried song
    return df.iloc[top_n_idx][['track_name', 'artist_names']]

df = pd.read_csv('./data/top-10k-spotify-songs-2025-07-detailed.csv')
print(recommend(df, 'Firework', 10))