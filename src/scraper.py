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

def cleanLyrics(text):
    text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    text = text.replace('\n', ' ').replace('-', ' ').replace('—', ' ')
    text = re.sub(r"[^a-zA-Z ]", "", text.lower())
    return text

def findLyrics(artist, title):
    time.sleep(2)
    song = genius.search_song(title=title, artist=artist)
    if song:
        return cleanLyrics(song.lyrics)

    if "/" in title:
        parts = [part.strip() for part in title.split(" / ")]
        songlist = []
        for part in parts:
            time.sleep(1)
            songEl = genius.search_song(title=part, artist=artist)
            if songEl:
                songlist.append(songEl.lyrics)
        return cleanLyrics('\n'.join(songlist))
        
    return None

# drakey = pd.read_csv('./data/drake.csv')
# titles = drakey['title']
# lyrics = [findLyrics('Drake', title) for title in titles]
# drakey['lyrics'] = lyrics
# drakey.to_csv('./data/drake.csv', index=False)


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
            driver.quit()
            if container:
                bpm_text = container.text.strip()
                if bpm_text.isdigit():
                    return int(bpm_text)

    print("BPM not found.")
    return None

# data = pd.read_csv('./data/drake.csv')
# bpms = [findBPM(artist.split(';')[0], title) for artist, title in zip(data['artist'], data['title'])]
# print(bpms)

def browse_artist_releases(artist : str, limit : int, offset : int, filepath: str = './data/artist_data.csv'):
    data = pd.DataFrame(columns=['title', 'artist', 'genre', 'year', 'album', 'bpm', 'lyrics'])
    if os.path.isfile(filepath) and all(pd.read_csv(filepath).columns == data.columns):
        pass
    else:
        data.to_csv(filepath, index=False)
    
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

        releases = mb.get_release_group_by_id(group['id'], includes=['releases'])[
            'release-group']['release-list']
        first_id = releases[0]['id']
        release_id = first_id
        for release in releases:
            if 'country' in release and release['country'] == 'XW':
                release_id = release['id']
                break
        if release_id == first_id:
            for release in releases:
                if 'country' in release and release['country'] == 'US':
                    release_id = release['id']
                    break

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

        bpms = [findBPM(artist, title) for title in titles] # dont wanna waste lyric requests if bpm fails
        lyrics = [findLyrics(artist, title) for title in titles]
        album = pd.DataFrame([
            [title, artist, genres, date[:4], album_title, bpm, lyric]
            for title, artist, bpm, lyric in zip(titles, artists, bpms, lyrics)
        ], columns=data.columns)
        album.to_csv(filepath, mode='a', header=False, index=False)
        print(album)

# browse_artist_releases('Drake', 5, 5, './data/drake.csv')

# Title Artist Genre Year BPM Lyrics

# Lyrics (semantic similarity: BERT, TF-IDF, or Doc2Vec)
# Acoustic features: BPM, key, mode, energy, valence (e.g., Spotify or audio analysis)
# (Optional later) Add genre and artist to re-rank or boost known favorites, but only after the feel-based matching works.



# Figure out if can get first release year from recording, decide on if genre is necessary, then pivot to lyrics/bpm/etc
# genre and release year are not that important rn
