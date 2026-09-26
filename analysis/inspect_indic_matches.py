import os
import re

base_dir = r"c:\VACSYN\student_resource\dataset\train"
gt_path = os.path.join(base_dir, "train_ground_truth.tsv")
s1_path = os.path.join(base_dir, "train_source1.tsv")
s2_path = os.path.join(base_dir, "train_source2.tsv")
s3_path = os.path.join(base_dir, "train_source3.tsv")

# We want to find examples in GT where S2 or S3 is non-ASCII
# Let's inspect the first 100,000 rows of GT, find non-ASCII S2 or S3 IDs, and look them up!
non_ascii_pattern = re.compile(r'[^\x00-\x7F]')

# First, find some non-ascii records in S2
s2_indic_samples = {}
with open(s2_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        eid, name, addr, cty = line.rstrip("\r\n").split("\t")
        if non_ascii_pattern.search(name) or non_ascii_pattern.search(addr):
            s2_indic_samples[eid] = (name, addr, cty)
            if len(s2_indic_samples) >= 1000:
                break

print(f"Found {len(s2_indic_samples)} S2 records with non-ASCII text.")

# Now check if any of these are in ground truth!
s1_to_s2_match = {}
with open(gt_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        s1, _, rest = line.rstrip("\r\n").partition("\t")
        if rest:
            for mid in rest.split(","):
                if mid in s2_indic_samples:
                    s1_to_s2_match[s1] = (mid, s2_indic_samples[mid])
                    if len(s1_to_s2_match) >= 10:
                        break
        if len(s1_to_s2_match) >= 10:
            break

print(f"Found {len(s1_to_s2_match)} matches with non-ASCII S2 records!")

# Now lookup S1 records
s1_lookup = {}
with open(s1_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        s1, name, addr, cty = line.rstrip("\r\n").split("\t")
        if s1 in s1_to_s2_match:
            s1_lookup[s1] = (name, addr, cty)

for s1, (s2, s2_data) in s1_to_s2_match.items():
    s1_data = s1_lookup.get(s1, ("?", "?", "?"))
    print("=" * 60)
    print(f"S1: ID={s1} | Name='{s1_data[0]}' | Addr='{s1_data[1]}' | Country={s1_data[2]}")
    print(f"S2: ID={s2} | Name='{s2_data[0]}' | Addr='{s2_data[1]}' | Country={s2_data[2]}")
