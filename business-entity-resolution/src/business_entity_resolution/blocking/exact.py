"""
Deterministic exact blocking key generation and inverted index matching.

Responsible for:
- Generating exact blocking keys partitioned strictly by country:
  * country + normalized name
  * country + name prefix (first 5 characters)
  * country + house/building number
  * country + distinctive address tokens
  * country + house number + name prefix (composite key)
- Managing inverted index lookups without Cartesian products.
- Guarding against bucket explosion by filtering oversized generic buckets.
- Providing high-level AddressBlocker integrating multiple address paths.

Ownership: Member 1 (Preprocessing & Blocking).
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Default maximum bucket size to prevent common terms from blowing up candidate sets
DEFAULT_MAX_BUCKET_SIZE = 500


def generate_exact_name_key(normalized_name: str, country: str) -> str:
    """
    Generate composite exact normalized name + country blocking key.

    Args:
        normalized_name: Cleaned business name.
        country: Country identifier string.

    Returns:
        Exact name blocking key (e.g. 'US::name::acme clinic'), or empty string if name is missing.
    """
    if not normalized_name or not country:
        return ""
    c = country.strip().upper()
    n = normalized_name.strip()
    return f"{c}::name::{n}" if n else ""


def generate_name_prefix_key(normalized_name: str, country: str, prefix_len: int = 5) -> str:
    """
    Generate business name prefix + country blocking key.

    Args:
        normalized_name: Cleaned business name.
        country: Country identifier string.
        prefix_len: Number of initial characters to include.

    Returns:
        Prefix blocking key (e.g. 'US::pref5::acme'), or empty string if name is missing.
    """
    if not normalized_name or not country:
        return ""
    c = country.strip().upper()
    n = normalized_name.strip()
    pref = n[:prefix_len] if len(n) >= prefix_len else n
    return f"{c}::pref{prefix_len}::{pref}" if pref else ""


def generate_house_number_key(house_number: Optional[str], country: str) -> Optional[str]:
    """
    Generate house number + country blocking key.

    Args:
        house_number: Extracted house/building number string.
        country: Country identifier string.

    Returns:
        House number blocking key (e.g. 'US::house::1795'), or None if house number is missing.
    """
    if not house_number or not country:
        return None
    c = country.strip().upper()
    h = house_number.strip().lower()
    return f"{c}::house::{h}" if h else None


def generate_address_token_keys(address_tokens: Iterable[str], country: str) -> List[str]:
    """
    Generate individual distinctive address token + country blocking keys.

    Args:
        address_tokens: List or set of address terms.
        country: Country identifier string.

    Returns:
        List of address token blocking keys (e.g. ['US::addr::westchester', 'US::addr::point']).
    """
    if not address_tokens or not country:
        return []
    c = country.strip().upper()
    keys = []
    for tok in address_tokens:
        t = tok.strip().lower()
        if t:
            keys.append(f"{c}::addr::{t}")
    return keys


def generate_house_number_name_prefix_key(
    house_number: Optional[str],
    normalized_name: str,
    country: str,
    prefix_len: int = 3,
) -> Optional[str]:
    """
    Generate composite house number + name prefix + country blocking key.
    Produces tight, discriminative buckets with high precision.

    Args:
        house_number: Extracted house/building number.
        normalized_name: Cleaned business name.
        country: Country identifier string.
        prefix_len: Number of characters of name prefix.

    Returns:
        Composite blocking key (e.g. 'US::h_pref3::1795::wes'), or None if house number is missing.
    """
    if not house_number or not normalized_name or not country:
        return None
    c = country.strip().upper()
    h = house_number.strip().lower()
    n = normalized_name.strip()
    pref = n[:prefix_len] if len(n) >= prefix_len else n
    return f"{c}::h_pref{prefix_len}::{h}::{pref}" if h and pref else None


class ExactIndex:
    """
    Inverted index for deterministic blocking keys with bucket-size capping.
    """

    def __init__(self, max_bucket_size: int = DEFAULT_MAX_BUCKET_SIZE) -> None:
        """
        Initialize inverted index.

        Args:
            max_bucket_size: Upper threshold on bucket size. Keys exceeding this
                             count are capped or excluded to prevent candidate explosion.
        """
        self.max_bucket_size = max_bucket_size
        self.index: Dict[str, List[str]] = defaultdict(list)

    def add(self, key: Optional[str], entity_id: str) -> None:
        """Add an entity ID under a given blocking key."""
        if key and entity_id:
            self.index[key].append(entity_id)

    def add_many(self, keys: Iterable[Optional[str]], entity_id: str) -> None:
        """Add an entity ID under multiple blocking keys."""
        for key in keys:
            if key and entity_id:
                self.index[key].append(entity_id)

    def get(self, key: Optional[str]) -> List[str]:
        """
        Retrieve candidate entity IDs for a key, applying bucket safety cap.

        Returns empty list if key is missing or exceeds max_bucket_size.
        """
        if not key or key not in self.index:
            return []
        bucket = self.index[key]
        if len(bucket) > self.max_bucket_size:
            # Drop or cap oversized generic buckets (e.g. words like 'hospital' or 'st')
            return []
        return bucket


def block_by_exact_name(
    normalized_name: str,
    country: str,
    name_index: ExactIndex,
) -> List[str]:
    """
    Retrieve candidate IDs sharing the exact normalized business name and country.
    """
    key = generate_exact_name_key(normalized_name, country)
    return name_index.get(key)


def block_by_name_prefix(
    normalized_name: str,
    country: str,
    prefix_index: ExactIndex,
    prefix_len: int = 5,
) -> List[str]:
    """
    Retrieve candidate IDs sharing the first prefix_len characters of business name and country.
    """
    key = generate_name_prefix_key(normalized_name, country, prefix_len=prefix_len)
    return prefix_index.get(key)


def block_by_house_number(
    house_number: Optional[str],
    country: str,
    house_index: ExactIndex,
) -> List[str]:
    """
    Retrieve candidate IDs sharing the same house number and country.
    """
    key = generate_house_number_key(house_number, country)
    return house_index.get(key)


def block_by_address_tokens(
    address_tokens: Iterable[str],
    country: str,
    address_token_index: ExactIndex,
    max_candidates: int = 100,
) -> List[str]:
    """
    Retrieve candidate IDs sharing at least one distinctive address token in the same country.
    """
    keys = generate_address_token_keys(address_tokens, country)
    seen: Set[str] = set()
    result: List[str] = []

    for k in keys:
        candidates = address_token_index.get(k)
        for cid in candidates:
            if cid not in seen:
                seen.add(cid)
                result.append(cid)
                if len(result) >= max_candidates:
                    return result

    return result


def block_by_house_and_name_prefix(
    house_number: Optional[str],
    normalized_name: str,
    country: str,
    composite_index: ExactIndex,
    prefix_len: int = 3,
) -> List[str]:
    """
    Retrieve candidate IDs matching both house number and business name prefix.
    """
    key = generate_house_number_name_prefix_key(
        house_number=house_number,
        normalized_name=normalized_name,
        country=country,
        prefix_len=prefix_len,
    )
    return composite_index.get(key)


class AddressBlocker:
    """
    Composite address and house-number candidate blocker.

    Integrates:
    - Path 1: House Number + Name Prefix (tight, high precision).
    - Path 2: Distinctive Address Token overlap (catches street and locality matches).
    - Path 3: Standalone House Number (recovers matches when name is heavily corrupted).
    """

    def __init__(
        self,
        max_token_doc_freq: int = 500,
        max_candidates_per_entity: int = 50,
    ) -> None:
        self.max_candidates_per_entity = max_candidates_per_entity
        self.house_prefix_index = ExactIndex(max_bucket_size=max_token_doc_freq)
        self.house_index = ExactIndex(max_bucket_size=max_token_doc_freq)
        self.addr_token_index = ExactIndex(max_bucket_size=max_token_doc_freq)
        self.addr_token_freqs: Counter = Counter()

    def add_entity(
        self,
        entity_id: str,
        address: Optional[str],
        normalized_name: str,
        country: str,
        house_number: Optional[str] = None,
        address_tokens: Optional[Set[str]] = None,
    ) -> None:
        """
        Add an entity to the address blocking indices.
        """
        if not entity_id or not country:
            return

        # Extract components if not already provided
        if house_number is None and address:
            from ..preprocessing.address import extract_house_number
            house_number = extract_house_number(address)

        if address_tokens is None and address:
            from ..preprocessing.address import tokenize_address
            address_tokens = tokenize_address(address)

        # 1. House number + name prefix key
        if house_number and normalized_name:
            k_hp = generate_house_number_name_prefix_key(
                house_number, normalized_name, country, prefix_len=3
            )
            self.house_prefix_index.add(k_hp, entity_id)

        # 2. Standalone house number key
        if house_number:
            k_h = generate_house_number_key(house_number, country)
            self.house_index.add(k_h, entity_id)

        # 3. Distinctive address tokens
        if address_tokens:
            c = country.strip().upper()
            for tok in address_tokens:
                k_tok = f"{c}::addr::{tok.lower().strip()}"
                self.addr_token_index.add(k_tok, entity_id)
                self.addr_token_freqs[k_tok] += 1

    def query_candidates(
        self,
        address: Optional[str],
        normalized_name: str,
        country: str,
        house_number: Optional[str] = None,
        address_tokens: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Retrieve candidate IDs across all address blocking paths.
        Union results, deduplicate, and rank by match confidence.
        """
        if not country or (not address and not house_number):
            return []

        if house_number is None and address:
            from ..preprocessing.address import extract_house_number
            house_number = extract_house_number(address)

        if address_tokens is None and address:
            from ..preprocessing.address import tokenize_address
            address_tokens = tokenize_address(address)

        candidate_scores: Counter = Counter()

        # Path 1: House number + name prefix (strongest address signal, weight = 3)
        if house_number and normalized_name:
            cands_hp = block_by_house_and_name_prefix(
                house_number, normalized_name, country, self.house_prefix_index, prefix_len=3
            )
            for cid in cands_hp:
                candidate_scores[cid] += 3

        # Path 2: Distinctive address tokens (weight = 2 per shared distinctive token)
        if address_tokens:
            keys = generate_address_token_keys(address_tokens, country)
            # Prioritize rarer address tokens
            sorted_keys = sorted(keys, key=lambda k: self.addr_token_freqs.get(k, 0))
            for k in sorted_keys[:5]:
                for cid in self.addr_token_index.get(k):
                    candidate_scores[cid] += 2

        # Path 3: Standalone house number (weight = 1)
        if house_number:
            cands_h = block_by_house_number(house_number, country, self.house_index)
            for cid in cands_h:
                candidate_scores[cid] += 1

        top_candidates = [
            cid for cid, _ in candidate_scores.most_common(self.max_candidates_per_entity)
        ]
        return top_candidates
