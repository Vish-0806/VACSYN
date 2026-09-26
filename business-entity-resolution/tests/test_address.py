"""
Comprehensive unit tests for address normalization, tokenization, and house number extraction.

Covers:
- Normal English / US addresses
- Uppercase addresses and abbreviations expansion
- Reordered address components (e.g. City/State first)
- Missing, empty, and null addresses
- Addresses with diverse house number formats (US, India, France)
- Addresses without house numbers (landmark-based)
- Multilingual and Unicode address components
"""

import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.preprocessing.address import (
    normalize_address,
    extract_house_number,
    tokenize_address,
)


def test_normalize_basic_and_casing():
    # US standard
    addr = "1795 Westchester Drive, High Point, NC"
    norm = normalize_address(addr)
    assert norm == "1795 westchester drive high point nc"

    # Uppercase with abbreviation
    addr_upper = "105 ELM ST, MORGANTON, NC"
    assert normalize_address(addr_upper) == "105 elm street morganton nc"

    addr_upper2 = "914 PIERPONT AVE, CLEVELAND, OH"
    assert normalize_address(addr_upper2) == "914 pierpont avenue cleveland oh"


def test_abbreviation_expansion():
    # Various street types
    assert "road" in normalize_address("17560 Ellis Rd., Tahlequah, OK")
    assert "boulevard" in normalize_address("1200 Sunset Blvd, Los Angeles, CA")
    assert "suite" in normalize_address("500 Market St, Ste 400")
    assert "floor" in normalize_address("Tower 1, 2nd Fl, MG Road")


def test_reordered_components_and_separators():
    # Inverted order: State, City, Street
    addr1 = "OH, Columbus, 5559 Orville Avenue"
    norm1 = normalize_address(addr1)
    assert "5559 orville avenue" in norm1
    assert "columbus" in norm1

    # Complex Indian address with slashes and commas
    addr2 = "H.No.16-11-23/37/A, 2nd Floor, Sagar Hotel Building, Hyderabad, Telangana"
    norm2 = normalize_address(addr2)
    assert "16 11 23 37 a" in norm2
    assert "hyderabad" in norm2


def test_house_number_extraction_us():
    assert extract_house_number("1795 Westchester Drive, High Point, NC") == "1795"
    assert extract_house_number("105 ELM ST, MORGANTON, NC") == "105"
    assert extract_house_number("126-B New Line Road, Morristown, TN") == "126-b"
    # Reordered: number in subsequent segment
    assert extract_house_number("OH, Columbus, 5559 Orville Avenue") == "5559"


def test_house_number_extraction_india():
    # Explicit H.No
    assert extract_house_number("H.No.16-11-23/37/A, Hyderabad, Telangana") == "16-11-23/37/a"
    # Explicit Plot No
    assert extract_house_number("Plot No. 780, Khordha, Orissa") == "780"
    # Flat No
    assert extract_house_number("Flat No. 207, Sagar Building") == "207"
    # Leading number
    assert extract_house_number("797, Lake Town Block A, Kolkata") == "797"
    assert extract_house_number("2505, Tower 1, Oakwood") == "2505"
    assert extract_house_number("303, 3rd Floor Sakar 5") == "303"


def test_house_number_extraction_france():
    assert extract_house_number("175 Boulevard du Président Franklin Roosevelt, Bordeaux") == "175"
    assert extract_house_number("Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac") == "5"
    assert extract_house_number("20 Rue Parmentier, Dunkerque") == "20"
    assert extract_house_number("Lille, 329 Avenue de Dunkerque") == "329"


def test_addresses_without_house_numbers():
    # Landmark-only addresses without building numbers
    assert extract_house_number("Near SBI ATM, MG Road, Bengaluru, Karnataka") is None
    assert extract_house_number("Opposite Bus Stand, Main Market, Jaipur") is None
    assert extract_house_number("High Point, North Carolina") is None


def test_missing_and_empty_addresses():
    assert normalize_address(None) == ""
    assert normalize_address("") == ""
    assert normalize_address("    ") == ""
    assert normalize_address("---, , ///") == ""
    assert extract_house_number(None) is None
    assert extract_house_number("") is None
    assert tokenize_address(None) == set()
    assert tokenize_address("") == set()


def test_unicode_and_accents():
    french_addr = "15 Rue de l'Étoile, 75008 Paris, Île-de-France"
    norm = normalize_address(french_addr)
    assert "étoile" in norm
    assert "île de france" in norm

    indic_addr = "एम जी रोड, बेंगलुरु, कर्नाटक"
    norm_indic = normalize_address(indic_addr)
    assert "बेंगलुरु" in norm_indic


def test_tokenization():
    tokens = tokenize_address("2505, Tower 1, Runwal Greens, Near Fortis Hospital, Mumbai")
    # Generic stopwords like 'near', 'tower' should be excluded
    assert "near" not in tokens
    assert "tower" not in tokens
    # Distinctive locality and city tokens must be present
    assert "2505" in tokens
    assert "runwal" in tokens
    assert "greens" in tokens
    assert "fortis" in tokens
    assert "hospital" in tokens
    assert "mumbai" in tokens


if __name__ == "__main__":
    test_normalize_basic_and_casing()
    test_abbreviation_expansion()
    test_reordered_components_and_separators()
    test_house_number_extraction_us()
    test_house_number_extraction_india()
    test_house_number_extraction_france()
    test_addresses_without_house_numbers()
    test_missing_and_empty_addresses()
    test_unicode_and_accents()
    test_tokenization()
    print("All Stage 3 address normalization and house number extraction tests passed successfully!")
