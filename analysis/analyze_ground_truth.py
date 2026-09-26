import os
import json
from collections import Counter

base_dir = r"c:\VACSYN"
gt_path = os.path.join(base_dir, "student_resource", "dataset", "train", "train_ground_truth.tsv")

total_s1 = 0
match_count_dist = Counter()
total_matches = 0
s2_matches_count = 0
s3_matches_count = 0

has_only_s2 = 0
has_only_s3 = 0
has_both_s2_s3 = 0
zero_matches = 0

max_matches = 0
max_matches_s1 = None

examples_singleton = []
examples_one_match = []
examples_multi_match = []
examples_both_s2_s3 = []

with open(gt_path, "r", encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        total_s1 += 1
        s1_id, tab, rest = line.rstrip("\r\n").partition("\t")
        mids = rest.split(",") if rest.strip() else []
        k = len(mids)
        match_count_dist[k] += 1
        total_matches += k
        
        if k > max_matches:
            max_matches = k
            max_matches_s1 = (s1_id, mids)
            
        s2_in_m = any(m.startswith("S2-") for m in mids)
        s3_in_m = any(m.startswith("S3-") for m in mids)
        
        for m in mids:
            if m.startswith("S2-"):
                s2_matches_count += 1
            elif m.startswith("S3-"):
                s3_matches_count += 1
                
        if k == 0:
            zero_matches += 1
            if len(examples_singleton) < 5:
                examples_singleton.append(s1_id)
        elif k == 1:
            if len(examples_one_match) < 5:
                examples_one_match.append({"source1_id": s1_id, "matched": mids})
        else:
            if len(examples_multi_match) < 5:
                examples_multi_match.append({"source1_id": s1_id, "matched": mids})
                
        if s2_in_m and s3_in_m:
            has_both_s2_s3 += 1
            if len(examples_both_s2_s3) < 5:
                examples_both_s2_s3.append({"source1_id": s1_id, "matched": mids})
        elif s2_in_m and not s3_in_m:
            has_only_s2 += 1
        elif s3_in_m and not s2_in_m:
            has_only_s3 += 1

# Breakdown
dist_buckets = {
    "zero_matches": match_count_dist[0],
    "one_match": match_count_dist[1],
    "two_matches": match_count_dist[2],
    "three_matches": match_count_dist[3],
    "four_or_more": sum(match_count_dist[k] for k in match_count_dist if k >= 4)
}

dist_pcts = {
    k: round((v / total_s1) * 100, 4) for k, v in dist_buckets.items()
}

results = {
    "total_s1_entities": total_s1,
    "total_matches_across_all": total_matches,
    "average_matches_per_s1": round(total_matches / total_s1, 4),
    "max_matches_for_one_s1": max_matches,
    "max_matches_example": max_matches_s1,
    "match_count_distribution": dist_buckets,
    "match_count_distribution_percentage": dist_pcts,
    "detailed_histogram": dict(sorted(match_count_dist.items())),
    "matches_pointing_to_source2": s2_matches_count,
    "matches_pointing_to_source3": s3_matches_count,
    "s1_entities_with_both_s2_and_s3": has_both_s2_s3,
    "s1_entities_with_both_s2_and_s3_pct": round((has_both_s2_s3 / total_s1) * 100, 4),
    "s1_entities_with_only_s2": has_only_s2,
    "s1_entities_with_only_s2_pct": round((has_only_s2 / total_s1) * 100, 4),
    "s1_entities_with_only_s3": has_only_s3,
    "s1_entities_with_only_s3_pct": round((has_only_s3 / total_s1) * 100, 4),
    "examples_singleton": examples_singleton,
    "examples_one_match": examples_one_match,
    "examples_multi_match": examples_multi_match,
    "examples_both_s2_s3": examples_both_s2_s3
}

with open(r"c:\VACSYN\analysis\ground_truth_profile.json", "w", encoding="utf-8") as out:
    json.dump(results, out, indent=2)

print("Ground truth analysis complete!")
