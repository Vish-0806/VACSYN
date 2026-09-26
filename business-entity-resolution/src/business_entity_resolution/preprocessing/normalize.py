"""
Business name normalization utilities.

Responsible for:
- Unicode-aware normalization preserving Indic and accented characters.
- Case normalization.
- Punctuation handling without destroying Indic combining marks (matras, viramas).
- Whitespace normalization.
- Ampersand and symbol normalization ('&' -> ' and ').
- Legal suffix handling (Inc, LLC, Ltd, Pvt Ltd, SARL, SAS, etc.).
- Tokenization helpers.

Important:
    Do NOT use ASCII-only normalization that destroys Indic scripts
    (Devanagari, Tamil, Telugu, etc.) or French accents.

Ownership: Member 1 (Preprocessing & Blocking).
"""

import logging
import re
import unicodedata
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Single-token legal entity designations across US, India, UK, France
SINGLE_LEGAL_SUFFIXES = {
    # English / US / Commonwealth
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "llp",
    "lp",
    "plc",
    "pvt",
    "private",
    "gmbh",
    # French entity designations
    "sarl",
    "sas",
    "sasu",
    "sa",
    "eurl",
    "sci",
    "snc",
    "gie",
    # Indic script designations
    "लिमिटेड",
    "प्राइवेट",
    "प्रा",
    "लि",
}

# Multi-token legal entity phrases (ordered from longest to shortest)
MULTI_TOKEN_LEGAL_PHRASES = [
    ("private", "limited"),
    ("pvt", "ltd"),
    ("pvt", "limited"),
    ("co", "ltd"),
    ("co", "limited"),
    ("co", "inc"),
    ("co", "corp"),
    ("co", "llc"),
    ("soc", "anonyme"),
    ("प्राइवेट", "लिमिटेड"),
    ("प्रा", "लि"),
]

RE_AMPERSAND = re.compile(r"&")


def normalize_business_name(name: Optional[str], strip_legal: bool = False) -> str:
    """
    Normalize a business name preserving multilingual Unicode characters.

    Pipeline:
        1. Handle None / empty / non-string input.
        2. Unicode NFKC normalization (composes accented letters, resolves ligatures).
        3. Case normalization (lowercase).
        4. Standardize '&' to ' and '.
        5. Replace punctuation and symbols with spaces while strictly preserving
           all letters (L), combining marks/matras (M), and numbers (N).
        6. Collapse multiple whitespaces and strip boundaries.
        7. Optional: Strip trailing legal entity designations.

    Args:
        name: Raw business name string.
        strip_legal: If True, remove recognized legal entity suffixes at end of string.

    Returns:
        Cleaned, normalized business name string.
    """
    if name is None:
        return ""

    if not isinstance(name, str):
        name = str(name)

    # Unicode NFKC normalization: standardizes ligatures and full-width chars
    text = unicodedata.normalize("NFKC", name)

    # If the text contains no letters or digits across any language, treat as empty
    if not any(unicodedata.category(c)[0] in ("L", "N") for c in text):
        return ""

    # Lowercase
    text = text.lower()

    # Standardize conjunctions
    text = RE_AMPERSAND.sub(" and ", text)

    # Replace punctuation, symbols, and underscores with space while strictly preserving
    # letters (L), combining vowel marks/matras/viramas (M), and numbers (N).
    # In Indic scripts, vowels are combining marks (Mn, Mc) that must NOT be stripped!
    clean_chars = []
    for char in text:
        cat = unicodedata.category(char)
        if cat[0] in ("P", "S") or char == "_":
            clean_chars.append(" ")
        else:
            clean_chars.append(char)
    text = "".join(clean_chars)

    # Normalize whitespace
    tokens = text.split()
    if not tokens:
        return ""

    if strip_legal and len(tokens) > 1:
        tokens = _strip_trailing_legal_suffixes(tokens)

    return " ".join(tokens)


def _strip_trailing_legal_suffixes(tokens: List[str]) -> List[str]:
    """
    Helper to strip recognized trailing legal suffixes from a token list.
    Preserves at least one token, and avoids stripping when the entire name
    consists only of legal entity terms (e.g. 'Private Limited').
    """
    # If all tokens are legal designations, do not strip
    if all(t in SINGLE_LEGAL_SUFFIXES for t in tokens):
        return tokens

    current = list(tokens)
    modified = True

    while modified and len(current) > 1:
        modified = False

        # If all remaining tokens are legal designations, stop stripping
        if all(t in SINGLE_LEGAL_SUFFIXES for t in current):
            break

        # Check multi-token suffixes
        for phrase in MULTI_TOKEN_LEGAL_PHRASES:
            k = len(phrase)
            if len(current) > k and tuple(current[-k:]) == phrase:
                remaining = current[:-k]
                if any(t not in SINGLE_LEGAL_SUFFIXES for t in remaining):
                    current = remaining
                    modified = True
                    break

        if modified:
            continue

        # Check single-token suffixes
        if len(current) > 1 and current[-1] in SINGLE_LEGAL_SUFFIXES:
            remaining = current[:-1]
            if any(t not in SINGLE_LEGAL_SUFFIXES for t in remaining):
                current.pop()
                modified = True

    return current if current else tokens


def extract_legal_suffixes(name: Optional[str]) -> Set[str]:
    """
    Extract recognized legal suffixes from a business name.

    Args:
        name: Raw or normalized business name.

    Returns:
        Set of detected legal suffix strings (e.g. {'pvt', 'ltd'}, {'inc'}).
    """
    if not name:
        return set()

    norm_name = normalize_business_name(name, strip_legal=False)
    tokens = norm_name.split()
    detected = set()

    # Check multi-token phrases
    for phrase in MULTI_TOKEN_LEGAL_PHRASES:
        k = len(phrase)
        for i in range(len(tokens) - k + 1):
            if tuple(tokens[i : i + k]) == phrase:
                detected.add(" ".join(phrase))

    # Check single-token suffixes
    for token in tokens:
        if token in SINGLE_LEGAL_SUFFIXES:
            detected.add(token)

    return detected


def tokenize_name(name: Optional[str], min_length: int = 2) -> List[str]:
    """
    Tokenize a business name into distinctive terms.

    Args:
        name: Raw or normalized business name.
        min_length: Minimum character length for tokens.

    Returns:
        List of cleaned token strings.
    """
    if not name:
        return []

    normalized = normalize_business_name(name, strip_legal=False)
    return [token for token in normalized.split() if len(token) >= min_length]


def tokenize_business_name(name: Optional[str], min_length: int = 2) -> List[str]:
    """
    Tokenize a business name into distinctive terms (alias for tokenize_name).

    Args:
        name: Raw or normalized business name.
        min_length: Minimum character length for tokens.

    Returns:
        List of cleaned token strings.
    """
    return tokenize_name(name=name, min_length=min_length)
