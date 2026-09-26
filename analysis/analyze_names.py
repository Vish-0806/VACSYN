import os
import json
import re
import unicodedata
from collections import Counter, defaultdict

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

LEGAL_SUFFIXES = re.compile(
    r'\b(inc|incorporated|llc|ltd|limited|pvt|private|corp|corporation|co|company|gmbh|sa|sas|sarl|llp|lp|plc)\b',
    re.IGNORECASE
)

def normalize_basic(text):
    if not text:
        return ""
    # Unicode NFKD
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'&', ' and ', text)
    # Replace punctuation with space
    text = re.sub(r'[^a-z0-9]', ' ', text)
    return ' '.join(text.split())

def normalize_legal_stripped(norm_text):
    if not norm_text:
        return ""
    # Remove legal suffixes
    tokens = norm_text.split()
    while tokens and LEGAL_SUFFIXES.fullmatch(tokens[-1]):
        tokens.pop()
    return ' '.join(tokens) if tokens else norm_text

def analyze_names_in_file(name, fpath):
    print(f"Analyzing names in {name}...")
    raw_counts = Counter()
    norm_counts = Counter()
    legal_counts = Counter()
    
    # Store collapsing examples (norm -> set of raw)
    collapsing_samples = defaultdict(set)
    sample_mappings = []
    
    total_rows = 0
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for idx, line in enumerate(f):
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            bname = parts[1].strip() if len(parts) > 1 else ""
            
            raw_counts[bname] += 1
            
            norm = normalize_basic(bname)
            norm_counts[norm] += 1
            
            legal = normalize_legal_stripped(norm)
            legal_counts[legal] += 1
            
            # Collect sample mappings
            if idx < 5:
                sample_mappings.append({
                    "raw": bname,
                    "normalized": norm,
                    "legal_stripped": legal
                })
                
            # Track a small set of collapsing examples
            if len(collapsing_samples) < 500:
                if len(collapsing_samples[norm]) < 5:
                    collapsing_samples[norm].add(bname)

    # Find actual collapsed examples where > 1 raw names map to same normalized
    collapsed_examples = []
    for norm, raws in collapsing_samples.items():
        if len(raws) > 1:
            collapsed_examples.append({
                "normalized": norm,
                "raw_variants": list(raws)
            })
            if len(collapsed_examples) >= 10:
                break
                
    total_unique_raw = len(raw_counts)
    duplicate_raw_keys = sum(1 for c in raw_counts.values() if c > 1)
    duplicate_raw_rows = sum(c for c in raw_counts.values() if c > 1)
    
    total_unique_norm = len(norm_counts)
    duplicate_norm_keys = sum(1 for c in norm_counts.values() if c > 1)
    duplicate_norm_rows = sum(c for c in norm_counts.values() if c > 1)
    
    total_unique_legal = len(legal_counts)
    
    # Group size distribution for normalized names
    group_sizes = Counter()
    for c in norm_counts.values():
        if c == 1:
            group_sizes["size_1"] += 1
        elif 2 <= c <= 5:
            group_sizes["size_2_to_5"] += 1
        elif 6 <= c <= 10:
            group_sizes["size_6_to_10"] += 1
        elif 11 <= c <= 50:
            group_sizes["size_11_to_50"] += 1
        else:
            group_sizes["size_50_plus"] += 1
            
    top_raw = [{"name": n, "count": c} for n, c in raw_counts.most_common(10)]
    top_norm = [{"name": n, "count": c} for n, c in norm_counts.most_common(10)]

    return {
        "source": name,
        "total_rows": total_rows,
        "unique_raw_names": total_unique_raw,
        "duplicate_raw_names_count": duplicate_raw_keys,
        "rows_with_duplicate_raw_name": duplicate_raw_rows,
        "pct_rows_with_duplicate_raw_name": round((duplicate_raw_rows / total_rows) * 100, 3) if total_rows > 0 else 0,
        "top_10_frequent_raw_names": top_raw,
        "unique_normalized_names": total_unique_norm,
        "duplicate_normalized_names_count": duplicate_norm_keys,
        "rows_with_duplicate_norm_name": duplicate_norm_rows,
        "pct_rows_with_duplicate_norm_name": round((duplicate_norm_rows / total_rows) * 100, 3) if total_rows > 0 else 0,
        "top_10_duplicated_normalized_names": top_norm,
        "unique_legal_stripped_names": total_unique_legal,
        "normalized_group_size_distribution": dict(group_sizes),
        "sample_mappings": sample_mappings,
        "collapsing_examples": collapsed_examples
    }

all_name_results = {}
for name, fpath in source_files.items():
    all_name_results[name] = analyze_names_in_file(name, fpath)

with open(r"c:\VACSYN\analysis\names_profile.json", "w", encoding="utf-8") as out:
    json.dump(all_name_results, out, indent=2)

print("Name analysis complete and saved to names_profile.json")
