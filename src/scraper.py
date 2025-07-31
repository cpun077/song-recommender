import musicbrainzngs as mb
import numpy as np
import pandas as pd

mb.set_useragent("SongRecommender", "1.0")

songs = []

artist_query = mb.search_artists('Drake')
artist_id = artist_query['artist-list'][0]['id']

offset = 0
release_groups = mb.browse_release_groups(
    artist=artist_id,
    includes=['artist-credits', 'tags'],
    limit=2,
    offset=offset
)['release-group-list']

for group in release_groups:
    album_title = group['title']
    titles = []
    artists = []
    date = group['first-release-date']

    tags = group['tag-list']
    # concat as string not array to flatten
    genres = ",".join(tag['name'] for tag in tags)

    release_id = mb.get_release_group_by_id(group['id'], includes=['releases'])[
        'release-group']['release-list'][0]['id']
    recordings = mb.get_release_by_id(release_id, includes=['artists', 'recordings'])[
        'release']['medium-list']

    for disc in recordings:
        tracklist = disc['track-list']
        titles = np.append(
            titles, [song['recording']['title'] for song in tracklist])
        artists = np.append(artists, [
            ",".join(entry['artist']['name']
                     for entry in song['artist-credit']
                     if isinstance(entry, dict) and 'artist' in entry
                     )  # concat as string not array to flatten
            for song in tracklist
        ])
    songs.extend([
        [title, artist, genres, date[:4], album_title]
        for title, artist in zip(titles, artists)
    ])

data = pd.DataFrame(songs, columns=['title', 'artist', 'genre', 'year', 'album'])
print(data)
# Title Artist Genre Year BPM Lyrics
