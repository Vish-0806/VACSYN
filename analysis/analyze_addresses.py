import os
import json
import re
from collections import Counter

base_dir = r"c:\VACSYN"
dataset_dir = os.path.join(base_dir, "student_resource", "dataset")

source_files = {
    "train_source1": os.path.join(dataset_dir, "train", "train_source1.tsv"),
    "train_source2": os.path.join(dataset_dir, "train", "train_source2.tsv"),
    "train_source3": os.path.join(dataset_dir, "train", "train_source3.tsv"),
    "test_source1": os.path.join(dataset_dir, "test", "test_source1.tsv"),
    "test_source2": os.path.join(dataset_dir, "test", "test_source2.tsv"),
    "test_source3": os.path.join(dataset_dir, "test", "test_source3.tsv"),
}

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
}

INDIA_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "chandigarh", "puducherry", "jammu", "kashmir",
    "ap", "ts", "up", "mp", "tn", "ka", "mh", "wb", "gj", "rj", "pb", "hr", "kl"
}

RE_US_ZIP = re.compile(r'\b\d{5}(?:-\d{4})?\b')
RE_INDIA_PIN = re.compile(r'\b[1-9]\d{5}\b')
RE_FR_POSTAL = re.compile(r'\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b')
RE_HOUSE_NUM = re.compile(r'\b\d+[a-zA-Z]?\b')
RE_SEPARATORS = re.compile(r'([,;/|\n])')

ABBREVIATIONS = [
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "blvd", "boulevard",
    "ln", "lane", "ct", "court", "pkwy", "parkway", "ste", "suite", "apt", "apartment",
    "fl", "floor", "no", "plot", "opp", "opposite", "nr", "near", "sec", "sector",
    "bldg", "building", "hwy", "highway", "rue", "bd", "av", "all", "chemin", "route"
]

def normalize_addr_simple(addr):
    if not addr:
        return ""
    a = addr.lower()
    a = re.sub(r'[^a-z0-9]', ' ', a)
    return ' '.join(a.split())

def calculate_percentiles_from_hist(length_counter, total_count, percentiles=[25, 50, 75, 90, 95, 99]):
    if total_count == 0:
        return {f"p{p}": 0 for p in percentiles}
    sorted_lengths = sorted(length_counter.keys())
    res = {}
    cum = 0
    targets = {p: (p / 100.0) * total_count for p in percentiles}
    current_p_idx = 0
    p_keys = sorted(percentiles)
    for length in sorted_lengths:
        cum += length_counter[length]
        while current_p_idx < len(p_keys) and cum >= targets[p_keys[current_p_idx]]:
            res[f"p{p_keys[current_p_idx]}"] = length
            current_p_idx += 1
    while current_p_idx < len(p_keys):
        res[f"p{p_keys[current_p_idx]}"] = sorted_lengths[-1] if sorted_lengths else 0
        current_p_idx += 1
    return res

def analyze_addresses_in_file(name, fpath):
    print(f"Analyzing addresses in {name}...")
    by_country = {}
    
    # We will sample 100,000 rows to do detailed address structure analysis
    # while running basic stats (missing, lengths) over all rows.
    
    stats_overall = {
        "total_rows": 0,
        "missing_count": 0,
        "raw_counts": Counter(),
        "norm_counts": Counter(),
        "len_hist": Counter()
    }
    
    country_data = {
        "US": {"rows": 0, "missing": 0, "len_hist": Counter(), "has_zip": 0, "has_house_num": 0, "has_state": 0, "samples": [], "separators": Counter(), "abbrevs": Counter()},
        "India": {"rows": 0, "missing": 0, "len_hist": Counter(), "has_zip": 0, "has_house_num": 0, "has_state": 0, "samples": [], "separators": Counter(), "abbrevs": Counter()},
        "France": {"rows": 0, "missing": 0, "len_hist": Counter(), "has_zip": 0, "has_house_num": 0, "has_state": 0, "samples": [], "separators": Counter(), "abbrevs": Counter()}
    }

    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for idx, line in enumerate(f):
            stats_overall["total_rows"] += 1
            parts = line.rstrip("\r\n").split("\t")
            addr = parts[2].strip() if len(parts) > 2 else ""
            country = parts[3].strip() if len(parts) > 3 else "Unknown"
            
            c_entry = country_data.get(country)
            if not c_entry:
                country_data[country] = {"rows": 0, "missing": 0, "len_hist": Counter(), "has_zip": 0, "has_house_num": 0, "has_state": 0, "samples": [], "separators": Counter(), "abbrevs": Counter()}
                c_entry = country_data[country]
                
            c_entry["rows"] += 1
            
            if not addr:
                stats_overall["missing_count"] += 1
                c_entry["missing"] += 1
                continue
                
            l = len(addr)
            stats_overall["len_hist"][l] += 1
            c_entry["len_hist"][l] += 1
            
            # To conserve memory across 5M rows, sample raw/norm duplicate rates on a subset or hashed set
            if idx < 300000:
                stats_overall["raw_counts"][addr] += 1
                stats_overall["norm_counts"][normalize_addr_simple(addr)] += 1
            
            # Detailed structural checks
            # House number
            if RE_HOUSE_NUM.search(addr):
                c_entry["has_house_num"] += 1
                
            # Postal code
            if country == "US" and RE_US_ZIP.search(addr):
                c_entry["has_zip"] += 1
            elif country == "India" and RE_INDIA_PIN.search(addr):
                c_entry["has_zip"] += 1
            elif country == "France" and RE_FR_POSTAL.search(addr):
                c_entry["has_zip"] += 1
                
            # State info
            addr_lower = addr.lower()
            tokens = set(re.findall(r'\b[a-z0-9]+\b', addr_lower))
            
            if country == "US":
                # Check uppercase 2-letter tokens for US state
                raw_tokens = set(re.findall(r'\b[A-Z]{2}\b', addr))
                if raw_tokens.intersection(US_STATES):
                    c_entry["has_state"] += 1
            elif country == "India":
                if tokens.intersection(INDIA_STATES):
                    c_entry["has_state"] += 1
            elif country == "France":
                # France departments or regions
                if any(w in tokens for w in ["idf", "ile", "france", "paris", "rhone", "gironde"]):
                    c_entry["has_state"] += 1

            # Separators & abbreviations on sample
            if len(c_entry["samples"]) < 500:
                for sep in RE_SEPARATORS.findall(addr):
                    c_entry["separators"][sep] += 1
                for ab in ABBREVIATIONS:
                    if ab in tokens:
                        c_entry["abbrevs"][ab] += 1
                if len(c_entry["samples"]) < 10:
                    c_entry["samples"].append(addr)

    # Summarize country stats
    country_summary = {}
    for c, cinfo in country_data.items():
        if cinfo["rows"] == 0:
            continue
        valid_rows = cinfo["rows"] - cinfo["missing"]
        pcts = calculate_percentiles_from_hist(cinfo["len_hist"], valid_rows)
        country_summary[c] = {
            "total_rows": cinfo["rows"],
            "missing_count": cinfo["missing"],
            "missing_pct": round((cinfo["missing"] / cinfo["rows"]) * 100, 3),
            "valid_address_count": valid_rows,
            "has_postal_code_pct": round((cinfo["has_zip"] / valid_rows) * 100, 2) if valid_rows > 0 else 0,
            "has_house_number_pct": round((cinfo["has_house_num"] / valid_rows) * 100, 2) if valid_rows > 0 else 0,
            "has_state_or_region_pct": round((cinfo["has_state"] / valid_rows) * 100, 2) if valid_rows > 0 else 0,
            "length_percentiles": pcts,
            "top_separators": dict(cinfo["separators"].most_common(5)),
            "top_abbreviations": dict(cinfo["abbrevs"].most_common(10)),
            "sample_addresses": cinfo["samples"][:5]
        }

    # Overall sample duplicate stats
    sample_raw_total = sum(stats_overall["raw_counts"].values())
    sample_raw_dups = sum(1 for c in stats_overall["raw_counts"].values() if c > 1)
    sample_norm_total = sum(stats_overall["norm_counts"].values())
    sample_norm_dups = sum(1 for c in stats_overall["norm_counts"].values() if c > 1)

    return {
        "source": name,
        "total_rows": stats_overall["total_rows"],
        "missing_count": stats_overall["missing_count"],
        "missing_pct": round((stats_overall["missing_count"] / stats_overall["total_rows"]) * 100, 4),
        "sample_duplicate_analysis": {
            "sample_size": sample_raw_total,
            "unique_raw_addresses": len(stats_overall["raw_counts"]),
            "duplicate_raw_keys": sample_raw_dups,
            "unique_normalized_addresses": len(stats_overall["norm_counts"]),
            "duplicate_normalized_keys": sample_norm_dups,
            "estimated_raw_duplicate_pct": round((sample_raw_dups / len(stats_overall["raw_counts"])) * 100, 2) if stats_overall["raw_counts"] else 0,
            "estimated_norm_duplicate_pct": round((sample_norm_dups / len(stats_overall["norm_counts"])) * 100, 2) if stats_overall["norm_counts"] else 0
        },
        "by_country": country_summary
    }

all_addr_results = {}
for name, fpath in source_files.items():
    all_addr_results[name] = analyze_addresses_in_file(name, fpath)

with open(r"c:\VACSYN\analysis\addresses_profile.json", "w", encoding="utf-8") as out:
    json.dump(all_addr_results, out, indent=2)

print("Address analysis complete and saved to addresses_profile.json")
