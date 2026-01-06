"""Utility to read sentiments.json, organize entries, and (optionally) rate via OpenAI.
OpenAI 5.2 prompt:
generate seven sentences about the state of affairs globally starting from very positive (rate as 7) to very negative (rate as 1).
"""

import json
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, TypedDict

from prompts import build_prompts
try:
    # OpenAI SDK v1
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class SentimentEntry(TypedDict):
    sentiment: int
    heading: str
    text: str


def load_sentiments_json(json_path: Path) -> List[SentimentEntry]:
    """Load the sentiments JSON array from the given file path."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def organize_sentiments_by_score(
    entries: List[SentimentEntry],
) -> Dict[int, Dict[str, str]]:
    """
    Convert a list of sentiment entries into a dict keyed by the numeric score.

    Example output:
    {
        7: {"heading": "Very positive", "text": "..."},
        6: {"heading": "Mostly positive", "text": "..."},
        ...
    }
    """
    organized: Dict[int, Dict[str, str]] = {}
    for entry in entries:
        score = int(entry["sentiment"])
        organized[score] = {
            "heading": entry["heading"],
            "text": entry["text"],
        }
    return organized


def rate_sentiment_with_openai_api(
    text: str,
    model: str = "gpt-4o-mini",
    runtime: str = "cloud",
    lang: str = "en",
) -> int:
    """
    Call OpenAI to rate the sentiment of a Belgian Dutch text from 1 (very negative) to 7 (very positive).
    Returns an integer in range [1, 7].
    - runtime='cloud': use OpenAI cloud with OPENAI_API_KEY (and optional OPENAI_BASE_URL)
    - runtime='local': use a local OpenAI-compatible server at LOCAL_OPENAI_BASE_URL or http://localhost:8000/v1
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK is not available. Please install the 'openai' package >= 1.0.")

    if runtime == "cloud":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Environment variable OPENAI_API_KEY is not set for cloud runtime.")
        base_url = os.getenv("OPENAI_BASE_URL")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url) if base_url else OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        # Local runtime: default to a localhost OpenAI-compatible endpoint and a dummy or provided key
        local_base_url = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:8000/v1")
        local_api_key = os.getenv("OPENAI_API_KEY", os.getenv("LOCAL_OPENAI_API_KEY", "sk-local"))
        client = OpenAI(api_key=local_api_key, base_url=local_base_url)

    system_prompt, user_prompt = build_prompts(text=text, lang=lang)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content if response.choices else ""
    # print(content)

    match = re.search(r"\[(\d+)\]", content or "") #   looks for a number in square brackets, like [5], anywhere in content.
    if not match:
        raise ValueError(f"Could not parse rating from model output: {content!r}")

    rating = int(match.group(1))
    if rating < 1:
        return 1
    if rating > 7:
        return 7
    return rating


def rate_all_texts(
    organized: Dict[int, Dict[str, str]],
    runtime: str = "cloud",
    lang: str = "en",
) -> Dict[int, int]:
    """
    Iterate over the organized dict and call the OpenAI rater for each text.
    Returns a dict mapping the original sentiment key to the model's rating.
    """
    results: Dict[int, int] = {}
    for score, payload in organized.items():
        predicted = rate_sentiment_with_openai_api(
            payload["text"], model="gpt-4o-mini", runtime=runtime, lang=lang
        )
        results[score] = predicted
    return results


def main() -> None:
    """Read sentiments dataset and print the organized dict."""
    parser = argparse.ArgumentParser(
        description="Organize sentiments from a JSON dataset (English or Dutch)."
    )
    parser.add_argument(
        "--lang",
        choices=["en", "nl"],
        required=True,
        help="Dataset language: 'en' -> sentiments-in-english.json, 'nl' -> sentiments-in-dutch.json",
    )
    parser.add_argument(
        "--runtime",
        choices=["cloud", "local"],
        default="cloud",
        help="Inference runtime: 'cloud' uses OpenAI API; 'local' uses a localhost OpenAI-compatible server.",
    )
    args = parser.parse_args()

    filename = (
        "sentiments-in-english.json" if args.lang == "en" else "sentiments-in-dutch.json"
    )
    json_path = Path(__file__).parent / filename
    entries = load_sentiments_json(json_path)
    organized = organize_sentiments_by_score(entries)
    # print(json.dumps(organized, indent=2, ensure_ascii=False))

    # If OPENAI_API_KEY is set and SDK is available, also obtain model ratings.
    # For local runtime, a real OPENAI_API_KEY is not required.
    if OpenAI is not None and (os.getenv("OPENAI_API_KEY") or args.runtime == "local"):
        model_ratings = rate_all_texts(organized, runtime=args.runtime, lang=args.lang)
        print(json.dumps({"model_ratings": model_ratings}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


