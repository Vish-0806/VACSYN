import os
import json
import math
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

def profile_file(name, fpath):
    print(f"Profiling {name}...")
    file_size_bytes = os.path.getsize(fpath)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    
    total_rows = 0
    cols = []
    
    missing_counts = {"entity_id": 0, "business_name": 0, "business_address": 0, "country": 0}
    
    name_len_hist = Counter()
    name_total_len = 0
    name_count = 0
    name_min_len = 999999
    name_max_len = 0
    
    addr_len_hist = Counter()
    addr_total_len = 0
    addr_count = 0
    addr_min_len = 999999
    addr_max_len = 0
    
    country_counts = Counter()
    
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        cols = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        
        for line in f:
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            # Ensure 4 parts
            while len(parts) < 4:
                parts.append("")
                
            eid, bname, baddr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            
            # Missing checks
            if not eid:
                missing_counts["entity_id"] += 1
            if not bname:
                missing_counts["business_name"] += 1
            else:
                l = len(bname)
                name_len_hist[l] += 1
                name_total_len += l
                name_count += 1
                if l < name_min_len: name_min_len = l
                if l > name_max_len: name_max_len = l
                
            if not baddr:
                missing_counts["business_address"] += 1
            else:
                l = len(baddr)
                addr_len_hist[l] += 1
                addr_total_len += l
                addr_count += 1
                if l < addr_min_len: addr_min_len = l
                if l > addr_max_len: addr_max_len = l
                
            if not country:
                missing_counts["country"] += 1
                country_counts["MISSING"] += 1
            else:
                country_counts[country] += 1

    # String stats for name
    name_pcts = calculate_percentiles_from_hist(name_len_hist, name_count)
    name_stats = {
        "min_length": name_min_len if name_count > 0 else 0,
        "max_length": name_max_len,
        "mean_length": round(name_total_len / name_count, 2) if name_count > 0 else 0,
        "median_length": name_pcts.get("p50", 0),
        "percentiles": name_pcts
    }
    
    # String stats for address
    addr_pcts = calculate_percentiles_from_hist(addr_len_hist, addr_count)
    addr_stats = {
        "min_length": addr_min_len if addr_count > 0 else 0,
        "max_length": addr_max_len,
        "mean_length": round(addr_total_len / addr_count, 2) if addr_count > 0 else 0,
        "median_length": addr_pcts.get("p50", 0),
        "percentiles": addr_pcts
    }
    
    # Country stats
    country_dist = {}
    for c, cnt in country_counts.most_common():
        country_dist[c] = {
            "count": cnt,
            "percentage": round((cnt / total_rows) * 100, 3) if total_rows > 0 else 0
        }
        
    missing_rates = {}
    for col, mcnt in missing_counts.items():
        missing_rates[col] = {
            "missing_count": mcnt,
            "missing_percentage": round((mcnt / total_rows) * 100, 4) if total_rows > 0 else 0
        }

    return {
        "name": name,
        "file_size_mb": file_size_mb,
        "total_rows": total_rows,
        "total_columns": len(cols),
        "column_names": cols,
        "missing_values": missing_rates,
        "business_name_stats": name_stats,
        "business_address_stats": addr_stats,
        "country_distribution": country_dist
    }

profiles = {}
for name, fpath in source_files.items():
    profiles[name] = profile_file(name, fpath)

with open(r"c:\VACSYN\analysis\sources_profile.json", "w", encoding="utf-8") as out:
    json.dump(profiles, out, indent=2)

print("Sources profile complete and written to sources_profile.json")
