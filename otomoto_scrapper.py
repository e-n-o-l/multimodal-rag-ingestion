import io, json, re, requests
from bs4 import BeautifulSoup
from PIL import Image


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
}


def fetch_soup(url: str) -> BeautifulSoup | None:
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        return BeautifulSoup(res.text, "html.parser") if res.status_code == 200 else None
    except Exception as e:
        print(f"website loading error {url}: {e}")
        return None


def extract_ad_data(soup: BeautifulSoup) -> dict:
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return {}
    try:
        props = json.loads(script.string).get("props", {}).get("pageProps", {})
        ad_data = props.get("ad") or props.get("advert") or {}
        if not ad_data and "urqlState" in props:
            for val in props["urqlState"].values():
                if "data" in val:
                    parsed = json.loads(val["data"])
                    if "advert" in parsed:
                        return parsed["advert"]
        return ad_data
    except Exception as e:
        print(f"JSON Error: {e}")
        return {}


def extract_clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()
    return re.sub(r"\n\s*\n", "\n", soup.get_text(separator="\n", strip=True))


def parse_photo_urls(ad_data: dict, max_photos: int = 30) -> list[str]:
    raw_photos = ad_data.get("images", {}).get("photos", [])
    urls = []
    for photo in raw_photos[:max_photos]:
        url = photo.get("full") or photo.get("large") or photo.get("url") if isinstance(photo, dict) else photo
        if url and url not in urls:
            urls.append(url)
    return urls


def download_images(photo_urls: list[str]) -> list[Image.Image]:
    images = []
    for url in photo_urls:
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                images.append(Image.open(io.BytesIO(res.content)).convert("RGB"))
        except Exception as e:
            print(f"photo loading error {url}: {e}")
    return images


def scrape_otomoto_offer(url: str, max_photos: int = 5) -> tuple[list[Image.Image], str] | None:
    soup = fetch_soup(url)
    if not soup:
        return None
    ad_data = extract_ad_data(soup)
    photo_urls = parse_photo_urls(ad_data, max_photos)
    return download_images(photo_urls), extract_clean_text(soup)


def get_offer_urls_from_page(page_num: int) -> list[str]:
    soup = fetch_soup(f"https://www.otomoto.pl/osobowe?page={page_num}")
    if not soup:
        return []
    return list(dict.fromkeys(
        a["href"] for a in soup.find_all("a", href=True) if "/oferta/" in a["href"]
    ))