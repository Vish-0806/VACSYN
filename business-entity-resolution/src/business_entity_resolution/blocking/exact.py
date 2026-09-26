"""
Exact blocking key generators.

Responsible for generating high-precision blocking keys:
- Normalized business name
- Country code (hard partition)
- Name prefix (first 3 to 5 characters)
- Extracted house/building numbers
- Distinctive address tokens

Ownership: Member 1 (Preprocessing & Blocking).
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def generate_exact_name_key(normalized_name: str, country: str) -> str:
    """
    Generate composite exact normalized name + country blocking key.

    Args:
        normalized_name: Cleaned business name.
        country: Country identifier string.

    Returns:
        Exact name blocking key.

    TODO:
        Implement key generation string formatting.
    """
    raise NotImplementedError("Exact name key generation is not implemented yet.")


def generate_name_prefix_key(normalized_name: str, country: str, prefix_len: int = 5) -> str:
    """
    Generate business name prefix + country blocking key.

    Args:
        normalized_name: Cleaned business name.
        country: Country identifier string.
        prefix_len: Number of initial characters to include.

    Returns:
        Prefix blocking key.

    TODO:
        Implement prefix key extraction.
    """
    raise NotImplementedError("Prefix key generation is not implemented yet.")


def generate_house_number_key(house_number: Optional[str], country: str) -> Optional[str]:
    """
    Generate house number + country blocking key.

    Args:
        house_number: Extracted house/building number string.
        country: Country identifier string.

    Returns:
        House number blocking key, or None if house number is missing.

    TODO:
        Implement house number key formatting.
    """
    raise NotImplementedError("House number key generation is not implemented yet.")


def generate_address_token_keys(address_tokens: List[str], country: str) -> List[str]:
    """
    Generate individual distinctive address token + country blocking keys.

    Args:
        address_tokens: List of address terms.
        country: Country identifier string.

    Returns:
        List of address token blocking keys.

    TODO:
        Implement address token key generation.
    """
    raise NotImplementedError("Address token keys generation is not implemented yet.")
