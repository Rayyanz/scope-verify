"""Text normalisation and containment helpers (SQuAD-style normalisation)."""
import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    """Lower-case, strip punctuation and articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch if ch not in _PUNCT else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def has(text: str, answer: str) -> bool:
    """True if the normalised answer occurs as a whole-token span in text."""
    a, t = normalize(answer), normalize(text)
    if not a:
        return False
    return re.search(r"(^| )" + re.escape(a) + r"( |$)", t) is not None


def in_evidence(paragraphs, answer: str) -> bool:
    """True if the answer occurs in any paragraph (dicts with 'title'/'text')."""
    return any(has(p.get("title", "") + " " + p["text"], answer) for p in paragraphs)
