"""Prompt builders for OpenAI sentiment rating."""

from typing import Tuple


def build_prompts(text: str, lang: str = "en") -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the requested language.
    - lang='nl' -> Dutch prompts
    - otherwise -> English prompts
    """
    if lang == "nl":
        system_prompt = (
            "Je bent een expert in de Nederlandse taal die de valentie van Belgisch-Nederlandse teksten analyseert."
        )
        user_prompt = (
            "Deelnemers reageerden op:\n"
            "‘Wat gebeurt er nu of sinds de vorige prompt, en hoe voel je je daarbij?’\n"
            f"Lees de reactie van de deelnemer zorgvuldig: {text}\n"
            "Jouw taak is om het sentiment te beoordelen van 1 (zeer negatief) tot 7 (zeer positief). "
            "Geef ALLEEN één numerieke beoordeling terug, ingesloten tussen vierkante haken, bijvoorbeeld [X], zonder aanvullende tekst.\n"
            "Uitvoerformaat: [getal]"
        )
    else:
        system_prompt = (
            "You are a Dutch language expert analyzing the valence of Belgian Dutch texts."
        )
        user_prompt = (
            "Participants responded to:\n"
            "‘What is going on now or since the last prompt, and how do you feel about it?’\n"
            f"Carefully read the response of the participant: {text}\n"
            "Your task is to rate its sentiment from 1 (very negative) to 7 (very positive).\n"
            "Return ONLY a single numerical rating enclosed in brackets, e.g. [X], with no additional text.\n"
            "Output Format: [number]"
        )

    return system_prompt, user_prompt


