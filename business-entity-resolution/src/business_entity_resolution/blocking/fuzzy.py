"""
Fuzzy string similarity and scalable name-token blocking utilities.

Responsible for:
- Token-set similarity blocking with frequency/IDF-style filtering.
- Recovering candidate pairs missed by exact matching while preventing bucket explosion.
- Character n-gram and edit similarity filtering for high-ambiguity entities.

Important:
    Generic tokens ('group', 'care', 'health', 'services', 'systems', 'pvt', 'inc')
    must be filtered or heavily down-weighted to prevent giant candidate buckets.

Ownership: Member 1 (Preprocessing & Blocking).
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Common stopwords and generic business descriptors that cause bucket explosions
GENERIC_NAME_TERMS = {
    # Conjunctions & prepositions
    "and",
    "the",
    "of",
    "in",
    "at",
    "for",
    "on",
    "a",
    "an",
    "to",
    "by",
    "is",
    "or",
    "de",
    "la",
    "le",
    "des",
    "du",
    "et",
    "en",
    "les",
    # Extremely frequent English/French/Indic trade descriptors
    "group",
    "services",
    "care",
    "health",
    "center",
    "centre",
    "management",
    "consulting",
    "solutions",
    "international",
    "enterprises",
    "holdings",
    "systems",
    "trading",
    "industries",
    "associates",
    "global",
    "national",
    "tech",
    "technology",
    "technologies",
    "clinic",
    "hospital",
    "club",
    "amicale",
    # Legal terms
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "pvt",
    "private",
    "sarl",
    "sas",
    "sasu",
    "sa",
    "eurl",
    "sci",
    "लिमिटेड",
    "प्राइवेट",
}


class NameTokenBlocker:
    """
    Scalable name token blocker with country partitioning and IDF-style bucket capping.
    """

    def __init__(
        self,
        max_token_doc_freq: int = 500,
        max_candidates_per_entity: int = 50,
    ) -> None:
        """
        Initialize name token blocker.

        Args:
            max_token_doc_freq: Maximum allowed bucket size for a single token.
                                Tokens appearing more frequently are suppressed to prevent explosion.
            max_candidates_per_entity: Maximum candidate budget returned per query entity.
        """
        self.max_token_doc_freq = max_token_doc_freq
        self.max_candidates_per_entity = max_candidates_per_entity
        self.inverted_index: Dict[str, List[str]] = defaultdict(list)
        self.token_freqs: Counter = Counter()

    def add_entity(self, entity_id: str, tokens: Iterable[str], country: str) -> None:
        """
        Add a candidate entity's distinctive name tokens to the country-partitioned index.
        """
        if not entity_id or not country:
            return

        c = country.strip().upper()
        unique_tokens = set(tokens)

        for tok in unique_tokens:
            t = tok.strip().lower()
            if len(t) >= 2 and t not in GENERIC_NAME_TERMS:
                key = f"{c}::{t}"
                self.inverted_index[key].append(entity_id)
                self.token_freqs[key] += 1

    def query_candidates(
        self,
        query_tokens: Iterable[str],
        country: str,
    ) -> List[str]:
        """
        Retrieve candidate IDs sharing distinctive name tokens with the query entity.

        Prioritizes the rarest (highest-IDF) tokens first to maximize precision
        while retrieving high-recall matches.
        """
        if not query_tokens or not country:
            return []

        c = country.strip().upper()
        # Filter distinctive query tokens
        valid_keys = []
        for tok in set(query_tokens):
            t = tok.strip().lower()
            if len(t) >= 2 and t not in GENERIC_NAME_TERMS:
                k = f"{c}::{t}"
                if k in self.inverted_index:
                    freq = self.token_freqs[k]
                    # Filter out oversized generic buckets
                    if freq <= self.max_token_doc_freq:
                        valid_keys.append((freq, k))

        if not valid_keys:
            return []

        # Sort by frequency ascending (rarest / most distinctive token first)
        valid_keys.sort(key=lambda x: x[0])

        candidate_scores: Counter = Counter()
        for freq, key in valid_keys:
            for cid in self.inverted_index[key]:
                candidate_scores[cid] += 1

        # Return candidates sorted by shared token count descending
        top_candidates = [
            cid for cid, _ in candidate_scores.most_common(self.max_candidates_per_entity)
        ]
        return top_candidates


def generate_token_overlap_candidates(
    query_tokens: Set[str],
    token_inverted_index: Dict[str, List[str]],
    min_shared_tokens: int = 1,
    max_candidates: int = 50,
) -> List[str]:
    """
    Retrieve candidate IDs from an inverted token index based on shared tokens.

    Args:
        query_tokens: Distinctive tokens for query entity.
        token_inverted_index: Mapping from token key to list of entity IDs.
        min_shared_tokens: Minimum required token overlaps.
        max_candidates: Upper limit on returned candidates to prevent bucket explosion.

    Returns:
        List of matching candidate entity IDs ranked by overlap count.
    """
    if not query_tokens or not token_inverted_index:
        return []

    overlap_counts: Counter = Counter()
    for tok in query_tokens:
        t = tok.strip().lower()
        if t in token_inverted_index:
            for cid in token_inverted_index[t]:
                overlap_counts[cid] += 1

    results = [
        cid
        for cid, count in overlap_counts.most_common(max_candidates)
        if count >= min_shared_tokens
    ]
    return results


def filter_candidates_by_edit_distance(
    query_text: str,
    candidate_ids: List[str],
    candidate_texts: Dict[str, str],
    threshold: float = 0.5,
) -> List[str]:
    """
    Filter a candidate list using fast fuzzy string matching (normalized Levenshtein / ratio).

    Args:
        query_text: Reference text string.
        candidate_ids: Pre-selected candidate IDs.
        candidate_texts: Mapping from candidate ID to text string.
        threshold: Minimum similarity ratio in [0.0, 1.0].

    Returns:
        Filtered list of candidate IDs with similarity >= threshold.
    """
    if not query_text or not candidate_ids:
        return []

    try:
        from rapidfuzz import fuzz

        use_rapidfuzz = True
    except ImportError:
        use_rapidfuzz = False

    filtered: List[str] = []
    q = query_text.strip().lower()

    for cid in candidate_ids:
        cand_text = candidate_texts.get(cid, "")
        if not cand_text:
            continue

        c = cand_text.strip().lower()
        if use_rapidfuzz:
            score = fuzz.ratio(q, c) / 100.0
        else:
            # Fallback character bigram dice similarity if rapidfuzz is unavailable
            score = _char_bigram_similarity(q, c)

        if score >= threshold:
            filtered.append(cid)

    return filtered


def _char_bigram_similarity(s1: str, s2: str) -> float:
    """Fallback character bigram Dice similarity."""
    if s1 == s2:
        return 1.0
    if len(s1) < 2 or len(s2) < 2:
        return 0.0
    b1 = Counter([s1[i : i + 2] for i in range(len(s1) - 1)])
    b2 = Counter([s2[i : i + 2] for i in range(len(s2) - 1)])
    overlap = sum((b1 & b2).values())
    total = sum(b1.values()) + sum(b2.values())
    return (2.0 * overlap) / total if total > 0 else 0.0
