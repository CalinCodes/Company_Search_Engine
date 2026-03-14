import os
import requests

_API_KEY = os.environ.get("GOOGLE_TRANSLATE_API_KEY")
_BASE_URL = "https://translation.googleapis.com/language/translate/v2"


def detect_language(text: str) -> str:
    """Returns a BCP-47 language code, e.g. 'en', 'fr', 'de'."""
    resp = requests.post(
        f"{_BASE_URL}/detect",
        params={"key": _API_KEY},
        json={"q": text},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["data"]["detections"][0][0]["language"]


def translate(texts: list[str], target: str, source: str = "en") -> list[str]:
    """Translates a list of strings. Returns translated strings in the same order."""
    if not texts:
        return []
    resp = requests.post(
        _BASE_URL,
        params={"key": _API_KEY},
        json={"q": texts, "target": target, "source": source, "format": "text"},
        timeout=15,
    )
    resp.raise_for_status()
    return [t["translatedText"] for t in resp.json()["data"]["translations"]]


def translate_results(results: list[dict], target_lang: str) -> list[dict]:
    """Translates the description field of each result back to the user's language."""
    descriptions = [r["company"].get("description") or "" for r in results]
    translated = translate(descriptions, target=target_lang)
    for result, desc in zip(results, translated):
        if desc:
            result["company"]["description"] = desc
    return results
