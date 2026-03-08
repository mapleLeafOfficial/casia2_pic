import urllib.request
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def search_kaggle():
    # Trying to hit Kaggle's public search API (might not be officially supported without auth but works for search sometimes, or we use a basic scraper)
    try:
        url = "https://www.kaggle.com/api/i/search/GetAllSearchQueries?query=image+forgery"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        print(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_kaggle()
