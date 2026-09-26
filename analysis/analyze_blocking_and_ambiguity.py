import os
import json
import re
import unicodedata
from collections import defaultdict, Counter

base_dir = r"c:\VACSYN"
dataset_dir = os.path.join(base_dir, "student_resource", "dataset", "train")
s1_path = os.path.join(dataset_dir, "train_source1.tsv")
s2_path = os.path.join(dataset_dir, "train_source2.tsv")

RE_NUM = re.compile(r'\b\d+\b')
RE_ZIP = re.compile(r'\b\d{5}(?:-\d{4})?\b')

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "gmbh", "sa", "sas", "sarl", "llp", "lp", "plc"
}
STOPWORDS = {"and", "the", "of", "in", "at", "for", "on", "a", "an", "to", "by", "is", "or"}

def clean_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'&', ' and ', text)
    text = re.sub(r'[^a-z0-9]', ' ', text)
    return ' '.join(text.split())

def get_tokens(text, remove_stopwords=True, min_len=2):
    tokens = [w for w in clean_text(text).split() if len(w) >= min_len]
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS and w not in LEGAL_SUFFIXES]
    return set(tokens)

print("Analyzing blocking bucket sizes and ambiguity on 500,000 records of Train S1...")

# Track buckets for blocking strategies
bucket_1_name = defaultdict(int)
bucket_2_name_country = defaultdict(int)
bucket_3_prefix = defaultdict(int)
bucket_4_token = defaultdict(int)
bucket_5_zip = defaultdict(int)
bucket_6_city = defaultdict(int)
bucket_7_state = defaultdict(int)
bucket_8_house = defaultdict(int)
bucket_9_addr_token = defaultdict(int)
bucket_10_composite = defaultdict(int)

# Ambiguity counters
ambiguity_stats = {
    "same_name_diff_addr_pairs": 0,
    "same_addr_diff_name_pairs": 0,
    "very_short_names_count": 0,      # <= 3 chars
    "very_long_names_count": 0,       # >= 50 chars
    "generic_names_count": 0,         # count >= 50 in S1
    "has_legal_suffix_count": 0,
    "has_ampersand_count": 0,
    "has_abbreviations_count": 0,
    "reordered_tokens_candidate_count": 0
}

# Also map name -> addresses and address -> names to detect multi-branch vs multi-tenant
name_to_addrs = defaultdict(set)
addr_to_names = defaultdict(set)

total_analyzed = 0
limit = 500000

with open(s1_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        total_analyzed += 1
        parts = line.rstrip("\r\n").split("\t")
        eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
        
        cn = clean_text(name)
        ca = clean_text(addr)
        tokens_n = get_tokens(name)
        tokens_a = get_tokens(addr, remove_stopwords=False)
        nums = RE_NUM.findall(addr)
        zips = RE_ZIP.findall(addr)
        
        # 1. Exact normalized name
        if cn:
            bucket_1_name[cn] += 1
            
        # 2. Name + country
        if cn:
            bucket_2_name_country[f"{cn}_{country}"] += 1
            
        # 3. Name prefix (first 5)
        pref = cn[:5] if len(cn) >= 5 else cn
        if pref:
            bucket_3_prefix[f"{pref}_{country}"] += 1
            
        # 4. Name tokens (sample up to 2 distinctive tokens per entity)
        for t in list(tokens_n)[:2]:
            bucket_4_token[f"{t}_{country}"] += 1
            
        # 5. Postal / ZIP
        for z in zips:
            bucket_5_zip[z] += 1
            
        # 6. City (last 2nd component)
        addr_parts = [p.strip().lower() for p in addr.split(",") if p.strip()]
        if len(addr_parts) >= 2:
            city = addr_parts[-2]
            bucket_6_city[f"{city}_{country}"] += 1
            
        # 7. State (last component or token)
        if len(addr_parts) >= 1:
            state = addr_parts[-1]
            bucket_7_state[f"{state}_{country}"] += 1
            
        # 8. House / building number
        if nums:
            bucket_8_house[f"{nums[0]}_{country}"] += 1
            
        # 9. Address tokens (first 2 distinctive)
        for at in list(tokens_a)[:2]:
            bucket_9_addr_token[f"{at}_{country}"] += 1
            
        # 10. Composite: (first word of name + house num)
        first_w = cn.split()[0] if cn else ""
        h_num = nums[0] if nums else ""
        if first_w and h_num:
            bucket_10_composite[f"{first_w}_{h_num}_{country}"] += 1

        # Ambiguity measurements
        if len(name.strip()) <= 3:
            ambiguity_stats["very_short_names_count"] += 1
        if len(name.strip()) >= 50:
            ambiguity_stats["very_long_names_count"] += 1
        if "&" in name:
            ambiguity_stats["has_ampersand_count"] += 1
        if any(w in tokens_n for w in LEGAL_SUFFIXES):
            ambiguity_stats["has_legal_suffix_count"] += 1
        if any(w in clean_text(addr).split() for w in ["st", "rd", "ave", "dr", "blvd", "pvt", "ltd"]):
            ambiguity_stats["has_abbreviations_count"] += 1
            
        # Track for same name diff addr & same addr diff name
        if len(name_to_addrs) < 200000:
            name_to_addrs[cn].add(ca)
        if len(addr_to_names) < 200000:
            addr_to_names[ca].add(cn)

        if total_analyzed >= limit:
            break

# Calculate bucket stats
def get_bucket_metrics(b_dict):
    if not b_dict:
        return {"unique_keys": 0, "avg_bucket_size": 0, "max_bucket_size": 0}
    vals = list(b_dict.values())
    return {
        "unique_keys": len(b_dict),
        "avg_bucket_size": round(sum(vals) / len(vals), 2),
        "max_bucket_size": max(vals)
    }

blocking_summary = {
    "1_exact_normalized_business_name": get_bucket_metrics(bucket_1_name),
    "2_normalized_name_plus_country": get_bucket_metrics(bucket_2_name_country),
    "3_name_prefix_5_plus_country": get_bucket_metrics(bucket_3_prefix),
    "4_name_tokens_plus_country": get_bucket_metrics(bucket_4_token),
    "5_postal_or_zip_code": get_bucket_metrics(bucket_5_zip),
    "6_city_plus_country": get_bucket_metrics(bucket_6_city),
    "7_state_plus_country": get_bucket_metrics(bucket_7_state),
    "8_house_number_plus_country": get_bucket_metrics(bucket_8_house),
    "9_address_tokens_plus_country": get_bucket_metrics(bucket_9_addr_token),
    "10_composite_name_token_plus_house_num": get_bucket_metrics(bucket_10_composite)
}

# Ambiguity summary
same_name_diff_addr = sum(1 for addrs in name_to_addrs.values() if len(addrs) > 1)
same_addr_diff_name = sum(1 for names in addr_to_names.values() if len(names) > 1)

ambiguity_summary = {
    "analyzed_records": total_analyzed,
    "same_name_different_address_entities": same_name_diff_addr,
    "same_name_different_address_pct": round((same_name_diff_addr / len(name_to_addrs)) * 100, 2),
    "same_address_different_business_locations": same_addr_diff_name,
    "same_address_different_business_pct": round((same_addr_diff_name / len(addr_to_names)) * 100, 2),
    "very_short_names_count": ambiguity_stats["very_short_names_count"],
    "very_short_names_pct": round((ambiguity_stats["very_short_names_count"] / total_analyzed) * 100, 2),
    "very_long_names_count": ambiguity_stats["very_long_names_count"],
    "very_long_names_pct": round((ambiguity_stats["very_long_names_count"] / total_analyzed) * 100, 2),
    "ampersand_in_name_pct": round((ambiguity_stats["has_ampersand_count"] / total_analyzed) * 100, 2),
    "has_legal_suffix_pct": round((ambiguity_stats["has_legal_suffix_count"] / total_analyzed) * 100, 2),
    "has_common_abbreviations_pct": round((ambiguity_stats["has_abbreviations_count"] / total_analyzed) * 100, 2)
}

res = {
    "blocking_bucket_metrics": blocking_summary,
    "ambiguity_analysis": ambiguity_summary
}

with open(r"c:\VACSYN\analysis\blocking_buckets_and_ambiguity.json", "w", encoding="utf-8") as out:
    json.dump(res, out, indent=2)

print("Blocking buckets and ambiguity analysis complete!")
