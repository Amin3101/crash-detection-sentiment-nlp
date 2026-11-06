"""Text pre-processing utilities.

Two stop-word lists are provided on purpose:

* ``MODEL_STOPWORDS``  - light; **keeps negations and intensifiers** ("not",
  "never", "very", "too" ...) because they flip / scale sentiment.
* ``CLOUD_STOPWORDS``  - aggressive; used only for word clouds, where we want
  content words and not grammar glue or generic e-commerce filler.
"""
from __future__ import annotations

import html
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that carry sentiment information and must survive stop-word removal.
SENTIMENT_KEEP = {
    "not", "no", "nor", "never", "none", "nothing", "nobody", "neither",
    "cannot", "without", "very", "too", "much", "more", "most", "less",
    "least", "few", "only", "enough", "against", "however", "but", "although",
    "though", "fully", "well", "again", "back", "off",
}
MODEL_STOPWORDS = frozenset(ENGLISH_STOP_WORDS - SENTIMENT_KEEP)

_GENERIC_FILLER = {
    "amazon", "product", "item", "one", "use", "used", "using", "get", "got",
    "would", "could", "also", "just", "bought", "buy", "purchased", "purchase",
    "ordered", "order", "review", "like", "really", "even", "make", "made",
    "thing", "things", "time", "will", "br", "href", "quot", "nbsp", "amp",
    # auxiliaries: once "not" is stripped, "did not work" would read as "did work"
    "did", "does", "doing", "done", "going", "went", "say", "said", "know", "think",
}
# NB: "work"/"works" are deliberately NOT filler - they are strongly informative
# ("not work", "works great").
CLOUD_STOPWORDS = frozenset(ENGLISH_STOP_WORDS | _GENERIC_FILLER)

# Bigrams that are pure negation "glue" ("does not", "did not" ...). They occur in every
# sentiment class, so for *word clouds* they only hide the informative phrases that follow.
NEGATION_GLUE_BIGRAMS = frozenset(
    f"{aux} not" for aux in (
        "does", "did", "do", "is", "was", "are", "were", "would", "could", "should",
        "will", "can", "have", "has", "had", "but", "and", "it", "that", "also", "just",
    )
)

_URL = re.compile(r"https?://\S+|www\.\S+")
_TAG = re.compile(r"<[^>]+>")
_NON_ALNUM = re.compile(r"[^a-z0-9!?'\s]")
_SPACES = re.compile(r"\s+")
_CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b|\bcannot\b"), "can not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'m\b"), " am"),
]


def clean_text(text: str) -> str:
    """Light normalisation that is safe for both linear models and transformers.

    Lower-cases, un-escapes HTML, drops URLs/tags, expands negative
    contractions (``don't`` -> ``do not``) and squeezes whitespace.
    Punctuation like ``!`` and ``?`` is preserved: it is a useful emotion cue.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).lower()
    text = _URL.sub(" ", text)
    text = _TAG.sub(" ", text)
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    text = _NON_ALNUM.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def sentiment_class(rating: int) -> str:
    """Map a 1-5 star rating to the three coarse classes used in Part 1."""
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"
