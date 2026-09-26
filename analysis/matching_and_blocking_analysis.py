import os
import json
import re
import unicodedata
from collections import defaultdict, Counter

base_dir = r"c:\VACSYN"
dataset_dir = os.path.join(base_dir, "student_resource", "dataset", "train")

gt_path = os.path.join(dataset_dir, "train_ground_truth.tsv")
s1_path = os.path.join(dataset_dir, "train_source1.tsv")
s2_path = os.path.join(dataset_dir, "train_source2.tsv")
s3_path = os.path.join(dataset_dir, "train_source3.tsv")

# Normalization functions
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "gmbh", "sa", "sas", "sarl", "llp", "lp", "plc"
}

STOPWORDS = {
    "and", "the", "of", "in", "at", "for", "on", "a", "an", "to", "by", "is", "or",
    "de", "la", "le", "des", "du", "et", "en", "les"
}

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

def jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def char_bigram_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    c1 = clean_text(s1)
    c2 = clean_text(s2)
    if not c1 or not c2:
        return 0.0
    if c1 == c2:
        return 1.0
    b1 = Counter([c1[i:i+2] for i in range(len(c1)-1)])
    b2 = Counter([c2[i:i+2] for i in range(len(c2)-1)])
    intersection = sum((b1 & b2).values())
    total = sum(b1.values()) + sum(b2.values())
    return (2.0 * intersection) / total if total > 0 else 0.0

# 1. Sample S1 entities from Ground Truth (sample 25,000 S1 entities that have matches)
print("Sampling ground truth pairs...")
sampled_pairs = [] # (s1_id, matched_id, source_type)
needed_s1 = set()
needed_s2 = set()
needed_s3 = set()

sample_limit = 25000
count_s1 = 0

with open(gt_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        s1_id, _, rest = line.rstrip("\r\n").partition("\t")
        mids = rest.split(",") if rest.strip() else []
        if mids:
            count_s1 += 1
            if count_s1 <= sample_limit:
                needed_s1.add(s1_id)
                for mid in mids:
                    stype = "S2" if mid.startswith("S2-") else "S3"
                    sampled_pairs.append((s1_id, mid, stype))
                    if stype == "S2":
                        needed_s2.add(mid)
                    else:
                        needed_s3.add(mid)
            else:
                break

print(f"Sampled {len(sampled_pairs)} true match pairs across {len(needed_s1)} S1 entities.")
print(f"Need {len(needed_s2)} S2 records and {len(needed_s3)} S3 records.")

# 2. Extract records via streaming
s1_records = {}
print("Loading needed S1 records...")
with open(s1_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        if parts[0] in needed_s1:
            s1_records[parts[0]] = {
                "name": parts[1] if len(parts) > 1 else "",
                "addr": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else ""
            }

s2_records = {}
print("Loading needed S2 records...")
with open(s2_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        if parts[0] in needed_s2:
            s2_records[parts[0]] = {
                "name": parts[1] if len(parts) > 1 else "",
                "addr": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else ""
            }

s3_records = {}
print("Loading needed S3 records...")
with open(s3_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        if parts[0] in needed_s3:
            s3_records[parts[0]] = {
                "name": parts[1] if len(parts) > 1 else "",
                "addr": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else ""
            }

# 3. Analyze Matching Difficulty
print("Calculating matching difficulty statistics...")
difficulty_stats = {
    "total_evaluated_pairs": len(sampled_pairs),
    "by_source": {"S2": {"count": 0}, "S3": {"count": 0}},
    "by_country": {"US": {"count": 0}, "India": {"count": 0}}
}

exact_norm_name_matches = 0
exact_norm_addr_matches = 0
name_jaccard_sum = 0.0
name_char_sim_sum = 0.0
name_len_diff_sum = 0
addr_jaccard_sum = 0.0
addr_char_sim_sum = 0.0
addr_len_diff_sum = 0
missing_addr_count = 0

identical_names = 0
similar_names = 0 # Jaccard >= 0.5
weak_name_strong_addr = 0 # name Jaccard < 0.3 and addr Jaccard >= 0.5
both_noisy = 0 # name Jaccard < 0.3 and addr Jaccard < 0.3

# Blocking Recall counters
blocking_hits = defaultdict(int)

# Extractors for blocking keys
RE_ZIP = re.compile(r'\b\d{5}(?:-\d{4})?\b')
RE_NUM = re.compile(r'\b\d+\b')

for s1_id, mid, stype in sampled_pairs:
    r1 = s1_records.get(s1_id)
    r2 = s2_records.get(mid) if stype == "S2" else s3_records.get(mid)
    if not r1 or not r2:
        continue
        
    n1, n2 = r1["name"], r2["name"]
    a1, a2 = r1["addr"], r2["addr"]
    c1, c2 = r1["country"], r2["country"]
    
    cn1, cn2 = clean_text(n1), clean_text(n2)
    ca1, ca2 = clean_text(a1), clean_text(a2)
    
    t_n1, t_n2 = get_tokens(n1), get_tokens(n2)
    t_a1, t_a2 = get_tokens(a1, remove_stopwords=False), get_tokens(a2, remove_stopwords=False)
    
    # Name stats
    is_exact_name = (cn1 == cn2) and (cn1 != "")
    if is_exact_name:
        exact_norm_name_matches += 1
        identical_names += 1
        
    n_jac = jaccard(t_n1, t_n2)
    n_sim = char_bigram_similarity(n1, n2)
    n_ldiff = abs(len(n1) - len(n2))
    name_jaccard_sum += n_jac
    name_char_sim_sum += n_sim
    name_len_diff_sum += n_ldiff
    
    if not is_exact_name and n_jac >= 0.5:
        similar_names += 1
        
    # Address stats
    is_addr_missing = not bool(a2.strip())
    if is_addr_missing:
        missing_addr_count += 1
        a_jac = 0.0
        a_sim = 0.0
        a_ldiff = len(a1)
    else:
        is_exact_addr = (ca1 == ca2) and (ca1 != "")
        if is_exact_addr:
            exact_norm_addr_matches += 1
        a_jac = jaccard(t_a1, t_a2)
        a_sim = char_bigram_similarity(a1, a2)
        a_ldiff = abs(len(a1) - len(a2))
        addr_jaccard_sum += a_jac
        addr_char_sim_sum += a_sim
        addr_len_diff_sum += a_ldiff
        
    if n_jac < 0.3 and a_jac >= 0.5:
        weak_name_strong_addr += 1
        
    if n_jac < 0.3 and a_jac < 0.3:
        both_noisy += 1
        
    # Test Blocking Strategies
    # 1. Exact normalized business name
    if cn1 and cn1 == cn2:
        blocking_hits["1_exact_norm_name"] += 1
        
    # 2. Normalized name + country
    if cn1 and cn1 == cn2 and c1 == c2:
        blocking_hits["2_norm_name_and_country"] += 1
        
    # 3. Name prefix (first 5 chars) + country
    pref1 = cn1[:5] if len(cn1) >= 5 else cn1
    pref2 = cn2[:5] if len(cn2) >= 5 else cn2
    if pref1 and pref1 == pref2 and c1 == c2:
        blocking_hits["3_name_prefix_5_and_country"] += 1
        
    # 4. Name tokens (at least 1 shared distinctive token) + country
    if (t_n1 & t_n2) and c1 == c2:
        blocking_hits["4_name_token_overlap_and_country"] += 1
        
    # 5. Postal / ZIP code
    z1 = RE_ZIP.findall(a1)
    z2 = RE_ZIP.findall(a2)
    if z1 and z2 and (set(z1) & set(z2)):
        blocking_hits["5_zip_code"] += 1
        
    # 6. City (token from last 2 components of address)
    parts_a1 = [p.strip().lower() for p in a1.split(",") if p.strip()]
    parts_a2 = [p.strip().lower() for p in a2.split(",") if p.strip()]
    city1 = parts_a1[-2] if len(parts_a1) >= 2 else ""
    city2 = parts_a2[-2] if len(parts_a2) >= 2 else ""
    if city1 and city1 == city2 and c1 == c2:
        blocking_hits["6_city_match_and_country"] += 1
        
    # 7. State / province (check state token match)
    if c1 == "US":
        s_tokens1 = set(re.findall(r'\b[A-Z]{2}\b', a1))
        s_tokens2 = set(re.findall(r'\b[A-Z]{2}\b', a2))
        if s_tokens1 and (s_tokens1 & s_tokens2):
            blocking_hits["7_state_and_country"] += 1
    elif c1 == "India":
        if (t_a1 & t_a2) and c1 == c2: # approximation
            blocking_hits["7_state_and_country"] += 1
            
    # 8. House / building number
    num1 = set(RE_NUM.findall(a1))
    num2 = set(RE_NUM.findall(a2))
    if num1 and num2 and (num1 & num2) and c1 == c2:
        blocking_hits["8_house_number_and_country"] += 1
        
    # 9. Address tokens (at least 1 shared address token) + country
    if (t_a1 & t_a2) and c1 == c2:
        blocking_hits["9_address_tokens_and_country"] += 1
        
    # 10. Multi-signal Union: (Name Token Overlap) OR (House Num + First Word of Name) OR (Exact Address + Country)
    first_w1 = cn1.split()[0] if cn1 else ""
    first_w2 = cn2.split()[0] if cn2 else ""
    cond_name_token = bool(t_n1 & t_n2) and (c1 == c2)
    cond_num_and_w1 = bool(num1 & num2) and (first_w1 == first_w2 and first_w1 != "") and (c1 == c2)
    cond_addr_token = (len(t_a1 & t_a2) >= 2) and (c1 == c2) and (n_sim >= 0.4)
    if cond_name_token or cond_num_and_w1 or cond_addr_token:
        blocking_hits["10_composite_multi_signal"] += 1

total_p = len(sampled_pairs)
valid_addr_pairs = total_p - missing_addr_count

difficulty_results = {
    "sample_pairs_evaluated": total_p,
    "name_similarity_metrics": {
        "exact_normalized_match_rate_pct": round((exact_norm_name_matches / total_p) * 100, 2),
        "mean_token_jaccard": round(name_jaccard_sum / total_p, 4),
        "mean_char_bigram_similarity": round(name_char_sim_sum / total_p, 4),
        "mean_length_difference": round(name_len_diff_sum / total_p, 2)
    },
    "address_similarity_metrics": {
        "missing_address_in_target_pct": round((missing_addr_count / total_p) * 100, 2),
        "exact_normalized_match_rate_pct": round((exact_norm_addr_matches / valid_addr_pairs) * 100, 2) if valid_addr_pairs > 0 else 0,
        "mean_token_jaccard": round(addr_jaccard_sum / valid_addr_pairs, 4) if valid_addr_pairs > 0 else 0,
        "mean_char_bigram_similarity": round(addr_char_sim_sum / valid_addr_pairs, 4) if valid_addr_pairs > 0 else 0,
        "mean_length_difference": round(addr_len_diff_sum / valid_addr_pairs, 2) if valid_addr_pairs > 0 else 0
    },
    "pair_categories": {
        "identical_normalized_names_pct": round((identical_names / total_p) * 100, 2),
        "similar_non_identical_names_pct": round((similar_names / total_p) * 100, 2),
        "weak_name_strong_address_pct": round((weak_name_strong_addr / total_p) * 100, 2),
        "both_name_and_address_noisy_pct": round((both_noisy / total_p) * 100, 2)
    },
    "blocking_candidate_recall": {}
}

for k, hits in blocking_hits.items():
    difficulty_results["blocking_candidate_recall"][k] = {
        "hits": hits,
        "total_pairs": total_p,
        "candidate_recall_pct": round((hits / total_p) * 100, 3)
    }

with open(r"c:\VACSYN\analysis\matching_and_blocking_profile.json", "w", encoding="utf-8") as out:
    json.dump(difficulty_results, out, indent=2)

print("Matching difficulty and blocking analysis complete!")
