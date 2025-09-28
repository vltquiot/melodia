import json, yaml, argparse, time, requests, os

BASE_URL = "https://api.discogs.com"
TOKEN = os.getenv("DISCOGS_TOKEN")

HEADERS = {
    "User-Agent": "MyMusicDataset/1.0",
    "Authorization": f"Discogs token={TOKEN}" if TOKEN else ""
}

def fetch_release(release_id):
    url = f"{BASE_URL}/releases/{release_id}"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        release = r.json()
    except requests.exceptions.HTTPError as e:
        print(f"/!\\Release {release_id} not found (404). Skipping.")
        return []

    tracks = []
    album_title = release["title"]
    genres = release.get("genres", [])
    styles = release.get("styles", [])
    year = release["year"]
    primary_artists = [a["name"] for a in release.get("artists", [])]
    country = release["country"]
    label = release["labels"][0]["name"] if release.get("labels") else None
    discogs_uri = release["uri"]
    
    for track in release.get("tracklist", []):
        track_data = {
            "track_title": track["title"],
            "primary_artists": primary_artists,
            "album_title": album_title,
            "year": year,
            "genres": genres,
            "styles": styles,
            "track_position": track["position"],
            "country": country,
            "label": label,
            "duration_sec": None,
            "discogs_uri": discogs_uri
        }

        # parse duration "mm:ss" -> int seconds
        dur = track["duration"]
        if dur and ":" in dur:
            m, s = dur.split(":")
            try:
                track_data["duration_sec"] = int(m) * 60 + int(s)
            except ValueError:
                pass

        tracks.append(track_data)

    return tracks

def fetch_releases_country(country="France", page=1, per_page=100):
    url = f"{BASE_URL}/database/search"
    params = {
        "type" : "release",
        "country" : country,
        "per_page" : per_page,
        "page" : page
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        r.raise_for_status()
        releases = r.json().get("results",[])
    except requests.exceptions.HTTPError as e:
        print(f"/!\\Request {params} failled (404). Skipping.")
        return False, []

    releases_ids = [release.get("id") for release in releases]

    return len(releases)==per_page, releases_ids


def main():
    cfg = yaml.safe_load(open('configs/ft_qlora.yaml'))
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=int(10e6))
    ap.add_argument("--output_file", type=str, default=cfg["dataset_discogs_path"])
    args = ap.parse_args()
    with open(args.output_file, "w", encoding="utf-8") as f:
        full_page = True
        page = 1
        while full_page and page*100<=args.n_samples:
            full_page, releases_ids = fetch_releases_country(page=page)
            time.sleep(1) # Max request for Discogs API 60/min -> 1/sec
            for release_id in releases_ids:
                tracks = fetch_release(release_id)
                for track_data in tracks: f.write(json.dumps(track_data, ensure_ascii=False) + "\n")
                time.sleep(1) # Max request for Discogs API 60/min -> 1/sec
            print(f"Wrote page {page}")
            page+=1
    
    print(f"Tracks data written to {args.output_file}")

if __name__ == "__main__":
   main()
