import json, requests, time, os
from pathlib import Path
from urllib.parse import quote

INPUT_FILE = "data/tracks.jsonl"
TRACKS_OUTPUT_DIR = "data/tracks_infos"
ARTISTS_OUTPUT_DIR = "data/artists_infos"
USER_AGENT = "Melodia/1.0 (Educational Project)" 

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

WIKI_USERNAME = "Vltquiot@Melodia"
WIKI_PASSWORD = "l3b9fs154j9gbd0rlhpl1p9pps4u7gjg"
USE_AUTHENTICATION = True

def get_login_token(session):
    params = {
        'action': 'query',
        'meta': 'tokens',
        'type': 'login',
        'format': 'json'
    }
    response = session.get(WIKI_API_URL, params=params)
    data = response.json()
    return data['query']['tokens']['logintoken']

def login(session):
    if not USE_AUTHENTICATION:
        return session
    
    print("\nAuthenticating with Wikipedia API...")
    
    login_token = get_login_token(session)

    login_params = {
        'action': 'login',
        'lgname': WIKI_USERNAME,
        'lgpassword': WIKI_PASSWORD,
        'lgtoken': login_token,
        'format': 'json'
    }
    
    response = session.post(WIKI_API_URL, data=login_params)
    data = response.json()
    
    if data['login']['result'] == 'Success':
        print("Authentication successful!")
        print(f"  Rate limit: 5,000 requests/hour")
        return session
    else:
        print(f"Authentication failed: {data['login']['result']}")
        print("  Falling back to unauthenticated requests (500/hour)")
        return session

def sanitize_filename(name):
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:200]

def get_wikipedia_content(title, search_type="song", session=None):

    if session is None:
        session = requests.Session()
        session.headers.update({'User-Agent': USER_AGENT})

    try:
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': title,
            'format': 'json',
            'srlimit': 1
        }
        
        response = session.get(WIKI_API_URL, params=search_params)
        response.raise_for_status()
        
        search_data = response.json()
        
        if not search_data['query']['search']:
            print(f"No Wikipedia page found for: {title}")
            return None
        
        page_title = search_data['query']['search'][0]['title']
    
        content_params = {
            'action': 'query',
            'prop': 'extracts|info',
            'titles': page_title,
            'format': 'json',
            'explaintext': True,
            'inprop': 'url'
        }
        
        time.sleep(0.6)
        
        response = session.get(WIKI_API_URL, params=content_params)
        response.raise_for_status()
        
        data = response.json()
        pages = data['query']['pages']
        page = next(iter(pages.values()))
        
        if 'extract' in page:
            content = f"Title: {page['title']}\n"
            content += f"URL: {page.get('fullurl', 'N/A')}\n"
            content += f"\n{page['extract']}\n"
            print(f"Found: {page['title']}")
            return content
        else:
            print(f"No content found for: {title}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {title}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for {title}: {e}")
        return None

def load_tracks(file_path):
    tracks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    return tracks

def save_content(content, output_dir, filename):
    if content:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

def fetch_tracks_info(tracks):
    print("\n" + "="*60)
    print("FETCHING TRACK INFORMATION")
    print("="*60)
    
    total = len(tracks)
    success_count = 0
    
    for idx, track in enumerate(tracks, 1):
        track_title = track.get('track_title', 'Unknown')
        artist = track.get('primary_artists', ['Unknown'])[0] if track.get('primary_artists') else 'Unknown'
        
        search_query = f"{track_title} {artist} song"
        
        print(f"\n[{idx}/{total}] Fetching: {track_title} by {artist}")
        
        content = get_wikipedia_content(search_query, "song")
        
        if content:
            filename = sanitize_filename(f"{track_title}.txt")
            save_content(content, TRACKS_OUTPUT_DIR, filename)
            success_count += 1
    
    print(f"\nTracks complete: {success_count}/{total} successful")

def fetch_artists_info(tracks):
    print("\n" + "="*60)
    print("FETCHING ARTIST INFORMATION")
    print("="*60)
    
    artists = set()
    for track in tracks:
        if track.get('primary_artists'):
            for artist in track['primary_artists']:
                artists.add(artist)
    
    artists = sorted(list(artists))
    total = len(artists)
    success_count = 0
    
    for idx, artist in enumerate(artists, 1):
        print(f"\n[{idx}/{total}] Fetching: {artist}")
        
        search_query = f"{artist} musician"
        content = get_wikipedia_content(search_query, "artist")
        
        if content:
            filename = sanitize_filename(f"{artist}.txt")
            save_content(content, ARTISTS_OUTPUT_DIR, filename)
            success_count += 1
    
    print(f"\nArtists complete: {success_count}/{total} successful")

def main():
    print("="*60)
    print("WIKIPEDIA MUSIC DATA FETCHER")
    print("="*60)
    
    print(f"\nLoading tracks from: {INPUT_FILE}")
    try:
        tracks = load_tracks(INPUT_FILE)
        print(f"Loaded {len(tracks)} tracks")
    except FileNotFoundError:
        print(f"Error: File not found: {INPUT_FILE}")
        return
    except Exception as e:
        print(f"Error loading tracks: {e}")
        return
    
    fetch_tracks_info(tracks)
    
    fetch_artists_info(tracks)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Track info saved to: {TRACKS_OUTPUT_DIR}/")
    print(f"Artist info saved to: {ARTISTS_OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
