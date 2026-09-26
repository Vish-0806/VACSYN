"""
Address preprocessing, normalization, and component extraction utilities.

Responsible for:
- Unicode-aware address standardization.
- Casing and whitespace normalization.
- Safe abbreviation expansion (St -> street, Rd -> road, Ave -> avenue, Blvd -> boulevard, etc.).
- Robust house / building / plot number extraction across US, India, and France formats.
- Address tokenization with generic term filtering for high-recall blocking.
- Graceful handling of missing or incomplete addresses (~3.3% in S2/S3).

Important:
    Do NOT assume postal/PIN codes are always available (0% in India, <0.4% in France, ~10.6% in US).
    Do NOT assume every address has a house number.
    Do NOT hardcode country rules that would break unseen France test data.

Ownership: Member 1 (Preprocessing & Blocking).
"""

import logging
import re
import unicodedata
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Safe street and location abbreviation mapping across English, Indian, and French datasets
STREET_ABBREVIATIONS = {
    # US & Commonwealth road types
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "dr": "drive",
    "blvd": "boulevard",
    "ln": "lane",
    "ct": "court",
    "pkwy": "parkway",
    "pl": "place",
    "ter": "terrace",
    "cir": "circle",
    "hwy": "highway",
    "fwy": "freeway",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    # French street abbreviations
    "av": "avenue",
    "bd": "boulevard",
    "all": "allee",
    "rte": "route",
    "ch": "chemin",
}

# Regex to standardize common prefixes
RE_PREFIX_HOUSE_NUM = re.compile(
    r"\b(?:h\.?\s*no\.?|house\s*no\.?|flat\s*no\.?|plot\s*no\.?|door\s*no\.?|sub\s*plot\s*no\.?|unit\s*no\.?|ews)\s*[-:#]?\s*([a-zA-Z0-9/-]+)",
    re.IGNORECASE,
)

# Regex to find leading numbers at the start of a comma-separated address segment
RE_SEGMENT_NUMBER = re.compile(
    r"^\s*([0-9]+(?:[/-][0-9a-zA-Z]+)?|[0-9]+[a-zA-Z]?)\b"
)

# Generic address noise terms that should not dominate blocking tokens
GENERIC_ADDRESS_STOPWORDS = {
    "near",
    "nr",
    "behind",
    "opp",
    "opposite",
    "next",
    "to",
    "beside",
    "adjacent",
    "at",
    "in",
    "on",
    "of",
    "and",
    "the",
    "floor",
    "fl",
    "nd",
    "th",
    "st",
    "rd",
    "phase",
    "sector",
    "sec",
    "block",
    "blk",
    "tower",
    "building",
    "bldg",
}


def normalize_address(address: Optional[str]) -> str:
    """
    Normalize an address string across US, India, and France formats.

    Pipeline:
        1. Handle None / empty / whitespace values gracefully.
        2. Unicode NFKC normalization.
        3. Lowercase.
        4. Expand common street abbreviations where safe.
        5. Normalize punctuation and separators to spaces while preserving
           letters, numbers, and combining marks.
        6. Collapse repeated whitespace and strip.

    Args:
        address: Raw address string or None/empty.

    Returns:
        Standardized lowercase address string, or empty string if missing.
    """
    if address is None:
        return ""

    if not isinstance(address, str):
        address = str(address)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", address).strip()
    if not text:
        return ""

    # Check if string contains any alphanumeric characters
    if not any(unicodedata.category(c)[0] in ("L", "N") for c in text):
        return ""

    text = text.lower()

    # Normalize separators: commas, slashes, hyphens
    text = text.replace("&", " and ")

    # Expand safe abbreviations on word boundaries
    tokens = text.split()
    expanded_tokens = []
    for tok in tokens:
        # Strip trailing punctuation on token for lookup
        clean_tok = tok.strip(".,;:-")
        if clean_tok in STREET_ABBREVIATIONS:
            expanded_tokens.append(STREET_ABBREVIATIONS[clean_tok])
        else:
            expanded_tokens.append(tok)
    text = " ".join(expanded_tokens)

    # Replace punctuation and symbols with spaces while keeping letters, marks, and digits
    clean_chars = []
    for char in text:
        cat = unicodedata.category(char)
        if cat[0] in ("P", "S") or char == "_":
            clean_chars.append(" ")
        else:
            clean_chars.append(char)
    text = "".join(clean_chars)

    # Collapse multiple whitespaces
    return " ".join(text.split())


def extract_house_number(address: Optional[str]) -> Optional[str]:
    """
    Extract primary house, building, plot, door, or unit number from an address.

    Handles varied regional conventions:
    - US: '1795 Westchester Drive', '5559 Orville Avenue', '126-B New Line Road'
    - India: 'H.No.16-11-23/37/A', 'Plot No. 780', 'Flat No. 207', '797, Lake Town'
    - France: '175 Boulevard...', '5 bis Rue Pierre Dignac', '30 Rue Lachassaigne'

    Args:
        address: Raw or normalized address string.

    Returns:
        Normalized alphanumeric number string (e.g. '1795', '16-11-23', '303', '5'),
        or None if no house/building number is identified.
    """
    if not address or not isinstance(address, str):
        return None

    raw = unicodedata.normalize("NFKC", address).strip()
    if not raw:
        return None

    # Strategy 1: Check explicit labeled number prefixes (H.No, Plot No, Flat No, Door No, EWS)
    labeled_match = RE_PREFIX_HOUSE_NUM.search(raw)
    if labeled_match:
        val = labeled_match.group(1).strip().lower().rstrip(".,;:-")
        # Ensure it contains at least one digit
        if any(c.isdigit() for c in val):
            return val

    # Strategy 2: Check leading number in comma-delimited segments
    # Useful when component order is inverted (e.g. 'OH, Columbus, 5559 Orville Avenue')
    segments = [seg.strip() for seg in raw.split(",") if seg.strip()]
    for seg in segments:
        seg_match = RE_SEGMENT_NUMBER.match(seg)
        if seg_match:
            candidate = seg_match.group(1).strip().lower().rstrip(".,;:-")
            if candidate and any(c.isdigit() for c in candidate):
                return candidate

    # Strategy 3: Check for first standalone numeric token in the address
    number_tokens = re.findall(r"\b\d+[a-zA-Z]?\b", raw)
    if number_tokens:
        return number_tokens[0].lower()

    return None


def tokenize_address(address: Optional[str], min_length: int = 2) -> Set[str]:
    """
    Tokenize an address into a set of distinctive geographic/street tokens.

    Filters out generic location noise words ('near', 'opposite', 'floor', etc.)
    so the resulting tokens serve as high-quality blocking keys.

    Args:
        address: Raw or normalized address string.
        min_length: Minimum character length for tokens.

    Returns:
        Set of distinctive address token strings.
    """
    if not address:
        return set()

    norm_addr = normalize_address(address)
    tokens = norm_addr.split()
    distinctive_tokens = set()

    for token in tokens:
        if len(token) >= min_length and token not in GENERIC_ADDRESS_STOPWORDS:
            distinctive_tokens.add(token)

    return distinctive_tokens
