import requests
import json

url = 'https://data.mendeley.com/public-api/datasets/kcmpdxky7p/versions/1'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("Fetching dataset info...")
r = requests.get(url, headers=headers)
if r.status_code == 200:
    data = r.json()
    files = data.get('files', [])
    print(f"Found {len(files)} files.")
    for f in files:
        print(f"File: {f['filename']} - Size: {f['size']} bytes")
        download_url = f.get('content_details', {}).get('download_url')
        if download_url:
            print(f"Download URL: {download_url}")
else:
    print(f"Failed to fetch. Status code: {r.status_code}")
