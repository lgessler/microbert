from bs4 import BeautifulSoup as BS
import requests as r
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

books = [
    "mat",
    "mar",
    "luk",
    "jon",
    "act",
    "rom",
    "co1",
    "co2",
    "gal",
    "eph",
    "phi",
    "col",
    "th1",
    "th2",
    "ti1",
    "ti2",
    "tit",
    "plm",
    "heb",
    "jam",
    "pe1",
    "pe2",
    "jo1",
    "jo2",
    "jo3",
    "jde",
    "rev",
]

urls = [f"https://www.sacred-texts.com/bib/wb/wlf/{book}.htm" for book in books]
outpaths = [BASE_DIR / (book + ".tsv") for book in books]

for url, outpath in zip(urls, outpaths):
    soup = BS(r.get(url).text, "html.parser")
    children = list(soup.find_all("p"))
    verses = []
    for child in children:
        try:
            anchor, text = list(child.children)
            verse_id = anchor.attrs["name"].split("_")[-1]
            verse = str(text).strip()
            verses.append((verse_id, verse))
        except ValueError:
            continue

    with open(outpath, 'w', encoding="utf8") as f:
        f.write("\n".join(("\t".join(verse) for verse in verses)) + "\n")
