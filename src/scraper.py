import musicbrainzngs as mb
import numpy as np
import pandas as pd

from lyricsgenius import Genius
from dotenv import load_dotenv
import os
import re

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

mb.set_useragent("SongRecommender", "1.0")

load_dotenv()
token = os.getenv('GENIUSTOKEN')
genius = Genius(token, sleep_time=1, timeout=10, retries=3, remove_section_headers=True)

options = Options()
options.add_argument('--user-agent=SongRecommender/1.0')

def findLyrics(artist, title):
    time.sleep(2)
    song = genius.search_song(title=title, artist=artist)
    return song.lyrics

def cleanLyrics(text):
    text = text.replace('\n', ' ').replace('-', ' ').replace('—', ' ')
    text = re.sub(r"[^a-zA-Z ]", "", text.lower())
    return text

import re

def cleanTitle(title):
    title = re.sub(r"[’']", "-", title)
    title = re.sub(r"[\"&|[\]().]", "", title)
    title = re.sub(r"\. |\s\(|\[", " ", title)
    title = re.sub(r"[\s\-]+", "-", title.strip())

    return title.lower()

def findBPM(artist, title):
    titleStr = cleanTitle(title)
    url = f'https://songbpm.com/@{artist}/{titleStr}'

    try:
        response = requests.get(url, timeout=1)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        container = soup.find(
            'dl', 'mt-5 grid grid-cols-1 gap-5 sm:grid-cols-3'
        ).find_all(
            'div', class_='bg-card overflow-hidden rounded-lg shadow-sm dark:border'
        )[2].find(
            'div', class_='hover:bg-secondary px-4 py-5 text-center sm:p-6'
        ).find(
            'dd', class_='text-card-foreground mt-1 text-3xl font-semibold'
        )

        if container:
            bpm_text = container.text.strip()
            if bpm_text.isdigit():
                return int(bpm_text)
    except requests.RequestException as e:
        print(e)
        print('Attempting crawling...')
        driver = webdriver.Chrome()
        driver.get('https://songbpm.com')
        time.sleep(2)
        search = driver.find_element(by=By.NAME, value='query')
        search.send_keys(f'{artist} {title}')
        search.submit()
        time.sleep(3)
        result = driver.find_element(
            by=By.CSS_SELECTOR, 
            value="a[class='hover:bg-foreground/5 flex flex-col items-start justify-start space-y-2 p-3 sm:flex-row sm:items-center sm:p-6']"
        )
        if result:
            result.click()
            time.sleep(2)
            container = driver.find_element(By.XPATH, "(//dd[@class='text-card-foreground mt-1 text-3xl font-semibold'])[3]")
            if container:
                bpm_text = container.text.strip()
                if bpm_text.isdigit():
                    return int(bpm_text)
        driver.quit()

    print("BPM not found.")
    return None

# data = pd.read_csv('./data/drake.csv')
# bpms = [findBPM(artist.split(';')[0], title) for artist, title in zip(data['artist'], data['title'])]
# print(bpms)

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
            'release-group']['release-list'][0]['id'] # last release maybe?
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
        bpms = [findBPM(artist, title) for title in titles]
        songs.extend([
            [title, artist, genres, date[:4], album_title, bpm, lyric]
            for title, artist, bpm, lyric in zip(titles, artists, bpms, lyrics)
        ])

    return pd.DataFrame(songs, columns=['title', 'artist', 'genre', 'year', 'album', 'bpm', 'lyrics'])

findings = browse_artist_releases('Drake', 10, 0) 
print(findings)
findings.to_csv('./data/drake.csv', index=False)

# Title Artist Genre Year BPM Lyrics

# Lyrics (semantic similarity: BERT, TF-IDF, or Doc2Vec)
# Acoustic features: BPM, key, mode, energy, valence (e.g., Spotify or audio analysis)
# (Optional later) Add genre and artist to re-rank or boost known favorites, but only after the feel-based matching works.



# Figure out if can get first release year from recording, decide on if genre is necessary, then pivot to lyrics/bpm/etc
# genre and release year are not that important rn
