"""
Comprehensive unit tests for business name normalization and tokenization.

Covers:
- Normal English names
- Punctuation variations and special symbols
- Uppercase/lowercase normalization
- '&' vs 'and' conjunction standardizations
- Legal entity suffix detection and stripping
- Repeated whitespace and boundary stripping
- Multilingual Indic scripts (Devanagari, Tamil, etc.) and French accents
- Empty, null, and non-string inputs
"""

import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.preprocessing.normalize import (
    normalize_business_name,
    tokenize_name,
    tokenize_business_name,
    extract_legal_suffixes,
)


def test_basic_and_casing():
    # Normal English
    assert normalize_business_name("Acme Corporation") == "acme corporation"
    assert normalize_business_name("ACME CORPORATION") == "acme corporation"
    assert normalize_business_name("acme corporation") == "acme corporation"


def test_punctuation_and_whitespace():
    # Repeated spaces, tabs, newlines
    assert normalize_business_name("  Acme   \t  Holdings \n  LLC  ") == "acme holdings llc"
    # Punctuation removal
    assert normalize_business_name("Orelee's Barbershop") == "orelee s barbershop"
    assert normalize_business_name("St. John's, Inc.!") == "st john s inc"
    assert normalize_business_name("Alpha / Beta - Gamma (Group)") == "alpha beta gamma group"


def test_ampersand():
    # & vs and
    assert normalize_business_name("Ear Nose & Throat Group") == "ear nose and throat group"
    assert normalize_business_name("AT&T Services") == "at and t services"
    assert normalize_business_name("Johnson & Johnson") == "johnson and johnson"


def test_indic_and_unicode_preservation():
    # Indic scripts must NOT be stripped or turned into empty strings!
    hindi_name = "राम मार्केटिंग प्राइवेट लिमिटेड"
    norm_hindi = normalize_business_name(hindi_name)
    assert norm_hindi == "राम मार्केटिंग प्राइवेट लिमिटेड"
    assert len(norm_hindi) > 0

    tamil_name = "ஆதித்யா பிராப்பர்ட்டீஸ் எல்எல்பி"
    norm_tamil = normalize_business_name(tamil_name)
    assert norm_tamil == "ஆதித்யா பிராப்பர்ட்டீஸ் எல்எல்பி"
    assert len(norm_tamil) > 0

    # French accents preserved and normalized
    french_name = "Café & Crêperie Étoile SARL"
    norm_french = normalize_business_name(french_name)
    assert norm_french == "café and crêperie étoile sarl"


def test_legal_suffix_handling():
    # Without strip_legal
    assert normalize_business_name("Summit Retail Inc", strip_legal=False) == "summit retail inc"
    
    # With strip_legal
    assert normalize_business_name("Summit Retail Inc", strip_legal=True) == "summit retail"
    assert normalize_business_name("Blue Horizon Pvt. Ltd.", strip_legal=True) == "blue horizon"
    assert normalize_business_name("Global Logistics Company Limited", strip_legal=True) == "global logistics"
    assert normalize_business_name("Marina Ecole France SARL", strip_legal=True) == "marina ecole france"
    
    # Preserves standalone legal names without making them empty
    assert normalize_business_name("Private Limited", strip_legal=True) == "private limited"

    # Legal suffix extraction
    suffixes = extract_legal_suffixes("ABC Technologies Pvt Ltd")
    assert "pvt ltd" in suffixes or ("pvt" in suffixes and "ltd" in suffixes)


def test_tokenization():
    tokens = tokenize_business_name("Apex Health Care, LLC", min_length=2)
    assert tokens == ["apex", "health", "care", "llc"]

    # Filter short tokens
    tokens_filtered = tokenize_name("A B C Health Care", min_length=2)
    assert tokens_filtered == ["health", "care"]

    # Multilingual tokenization
    indic_tokens = tokenize_business_name("सनराइज फूड्स प्राइवेट लिमिटेड", min_length=2)
    assert "सनराइज" in indic_tokens
    assert "फूड्स" in indic_tokens


def test_empty_and_null_values():
    assert normalize_business_name(None) == ""
    assert normalize_business_name("") == ""
    assert normalize_business_name("    ") == ""
    assert normalize_business_name("---") == ""
    assert normalize_business_name("!@#$%^&*()") == ""
    assert tokenize_business_name(None) == []
    assert tokenize_business_name("") == []
    assert extract_legal_suffixes(None) == set()


if __name__ == "__main__":
    test_basic_and_casing()
    test_punctuation_and_whitespace()
    test_ampersand()
    test_indic_and_unicode_preservation()
    test_legal_suffix_handling()
    test_tokenization()
    test_empty_and_null_values()
    print("All Stage 2 business name normalization tests passed successfully!")
