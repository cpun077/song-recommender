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
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    ElementClickInterceptedException, 
    ElementNotInteractableException, 
    TimeoutException, 
    WebDriverException, 
    StaleElementReferenceException
)
from selenium.webdriver.common.keys import Keys
import time

mb.set_useragent("SongRecommender", "1.0")

load_dotenv()
token = os.getenv('GENIUSTOKEN')
genius = Genius(token, sleep_time=1, timeout=10, retries=3, remove_section_headers=True)

useragent = os.getenv('USERAGENT')

options = Options()
options.add_argument('--user-agent=SongRecommender/1.0')
options.add_argument('--incognito')
options.page_load_strategy = 'normal'

def cleanLyrics(text):
    text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    text = text.replace('\n', ' ').replace('-', ' ').replace('—', ' ')
    text = re.sub(r"[^a-zA-Z ]", "", text.lower())
    return text

def findLyrics(artist, title):
    time.sleep(2)
    song = genius.search_song(title=title, artist=artist)
    if song:
        # print(song.lyrics)
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

def retryClick(element:str, driver:webdriver.Chrome, retries:int=3):
    for attempt in range(retries):
        try:
            wait = WebDriverWait(driver, 5)
            submit = wait.until(EC.element_to_be_clickable((
                By.CSS_SELECTOR,
                element
            )))
            driver.execute_script("arguments[0].scrollIntoView();", submit)
            try:
                print('click() initiated')
                submit.click()
            except (ElementClickInterceptedException, ElementNotInteractableException):
                print('script initiated')
                driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('click', {bubbles:true}))", submit)

            print('click succeeded')
            break
        except (TimeoutException, WebDriverException, StaleElementReferenceException) as e:
            print(f'Attempt {attempt + 1} failed: {e}')
            time.sleep(1)

def crawl(artist, title):
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 5)

    driver.get('https://songbpm.com')
    searchbar = wait.until(EC.element_to_be_clickable((By.TAG_NAME, "input")))
    searchbar.clear()
    old_url = driver.current_url
    searchbar.send_keys(f'{artist} {title}' + Keys.ENTER)
    # time.sleep(0.5)
    # print('return')
    # searchbar.send_keys(Keys.ENTER)
    wait.until(EC.url_changes(old_url))
    print('search results!')

    # retryClick("button[type='submit']", driver)

    retryClick("a[class='hover:bg-foreground/5 flex flex-col items-start justify-start space-y-2 p-3 sm:flex-row sm:items-center sm:p-6']", driver)
    
    # driver.execute_script("arguments[0].scrollIntoView(true);", result)
    # time.sleep(1)
    # driver.execute_script("arguments[0].click();", result)

    text = wait.until(EC.presence_of_element_located((
        By.XPATH,
        "(//dd[@class='text-card-foreground mt-1 text-3xl font-semibold'])[3]"
    ))).text.strip()

    print('BPM found')

    driver.quit()
    if text.isdigit():
        return int(text)
    return None

def songbpm(url:str):
    try:
        response = requests.get(url, timeout=1)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
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
        return None

def findBPM(artist:str, title:str):
    time.sleep(2)
    titleStr = cleanTitle(title)
    url = f'https://songbpm.com/@{artist}/{titleStr}'

    bpm = songbpm(url)
    if not bpm:
        # try:
        #     searchstr = f'{artist} {title} site:songbpm.com'
        #     googleurl = 'https://www.google.com/search'
        #     headers = {
        #         'Accept' : '*/*',
        #         'Accept-Language': 'en-US,en;q=0.5',
        #         'User-Agent': useragent,
        #     }
        #     parameters = {'q': searchstr}
        #     response = requests.get(googleurl, headers = headers, params = parameters)
        #     response.raise_for_status()
        #     soup = BeautifulSoup(response.text, 'html.parser')
        #     print(soup)
        #     atags = soup.find_all(href=True)
        #     links = []
        #     for a in atags:
        #         link = a.get('href')
        #         if f'https://songbpm.com/@{artist.lower()}' in link:
        #             links.append(link)
        #     print(links)
        #     firstlink = links[0]
        #     bpm = songbpm(firstlink)
        #     if bpm:
        #         return bpm
        # except requests.RequestException as e:
        #     print(e)
            return crawl(artist, title)
    else:
        return bpm

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

# browse_artist_releases('Drake', 4, 6, './data/drake.csv')

# Title Artist Genre Year BPM Lyrics

# Lyrics (semantic similarity: BERT, TF-IDF, or Doc2Vec)
# Acoustic features: BPM, key, mode, energy, valence (e.g., Spotify or audio analysis)
# (Optional later) Add genre and artist to re-rank or boost known favorites, but only after the feel-based matching works.



# Figure out if can get first release year from recording, decide on if genre is necessary, then pivot to lyrics/bpm/etc
# genre and release year are not that important rn
