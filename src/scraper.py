import musicbrainzngs as mb
import numpy as np
import pandas as pd

import requests
from lyricsgenius import Genius
from dotenv import load_dotenv
import os
import re

mb.set_useragent("SongRecommender", "1.0")

load_dotenv()
token = os.getenv('GENIUSTOKEN')
genius = Genius(token, sleep_time=1, timeout=10, retries=3, remove_section_headers=True)

def findLyrics(artist, title):
    song = genius.search_song(title=title, artist=artist)
    return song.lyrics

def cleanLyrics(text):
    text = text.replace('\n', ' ').replace('-', ' ').replace('—', ' ')
    text = re.sub(r"[^a-zA-Z ]", "", text.lower())
    return text

def browse_artist_releases(artist : str, limit : int, offset : int):
    songs = []
    artist_query = mb.search_artists(artist)
    artist_id = artist_query['artist-list'][0]['id']
    release_groups = mb.browse_release_groups(
        artist=artist_id,
        includes=['artist-credits', 'tags'],
        limit=limit,
        offset=offset
    )['release-group-list']

    for group in release_groups:
        album_title = group['title']
        titles = []
        artists = []
        date = group['first-release-date']

        tags = group['tag-list']
        # concat as string not array to flatten
        genres = ";".join(tag['name'] for tag in tags)

        release_id = mb.get_release_group_by_id(group['id'], includes=['releases'])[
            'release-group']['release-list'][0]['id']
        recordings = mb.get_release_by_id(release_id, includes=['artists', 'recordings'])[
            'release']['medium-list']

        for disc in recordings:
            tracklist = disc['track-list']
            titles = np.append(
                titles, [song['recording']['title'] for song in tracklist])
            artists = np.append(artists, [
                ";".join(entry['artist']['name']
                        for entry in song['artist-credit']
                        if isinstance(entry, dict) and 'artist' in entry
                        )  # concat as string not array to flatten
                for song in tracklist
            ])

        lyrics = [cleanLyrics(findLyrics(artist, title)) for title in titles]
        songs.extend([
            [title, artist, genres, date[:4], album_title, lyric]
            for title, artist, lyric in zip(titles, artists, lyrics)
        ])

    return pd.DataFrame(songs, columns=['title', 'artist', 'genre', 'year', 'album', 'lyrics'])

findings = browse_artist_releases('Drake', 1, 0) 
print(findings)
findings.to_csv('./data/drake.csv', index=False)

# def findLyrics(artist, title):
#     url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json().get('lyrics', '')
#     else:
#         print(f"Error: {response.status_code} for {artist} - {title}")
#         return ''

# Title Artist Genre Year BPM Lyrics

# Lyrics (semantic similarity: BERT, TF-IDF, or Doc2Vec)
# Acoustic features: BPM, key, mode, energy, valence (e.g., Spotify or audio analysis)
# (Optional later) Add genre and artist to re-rank or boost known favorites, but only after the feel-based matching works.



# Figure out if can get first release year from recording, decide on if genre is necessary, then pivot to lyrics/bpm/etc
# genre and release year are not that important rn
