import requests
from bs4 import BeautifulSoup
import pandas as pd
import random

pages = [
    "https://ar.wikipedia.org/wiki/الأمن_السيبراني",
    "https://ar.wikipedia.org/wiki/أمن_المعلومات",
    "https://ar.wikipedia.org/wiki/الاختراق_الحاسوبي",
    "https://ar.wikipedia.org/wiki/الهجمات_الإلكترونية",
    "https://ar.wikipedia.org/wiki/البرمجيات_الخبيثة",
    "https://ar.wikipedia.org/wiki/التشفير",
    "https://ar.wikipedia.org/wiki/جدار_ناري"
]

headers = {
    "User-Agent": "Mozilla/5.0"
}

texts = []

for url in pages:
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 60:
            texts.append(text)

print("Total textes collectés :", len(texts))

# Supprimer les doublons
texts = list(set(texts))
print("Après suppression des doublons :", len(texts))

# Dataset
N = min(500, len(texts))

df = pd.DataFrame({
    "Text": texts[:N],
    "Score": [round(random.uniform(0,10),1) for _ in range(N)]
})

df.to_csv("arabic_cybersecurity_texts.csv", index=False, encoding="utf-8-sig")
print("Dataset final sauvegardé ✅ :", len(df))
