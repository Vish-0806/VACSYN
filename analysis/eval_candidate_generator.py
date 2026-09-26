"""
Empirical evaluation of CandidateGenerator across multi-source training data.
Measures Recall, Candidate Count, and Contract 1 conformance.
"""

from collections import defaultdict
from pathlib import Path
import time
import pandas as pd

from business_entity_resolution.blocking.candidate_generator import CandidateGenerator


def evaluate():
    base_dir = Path("c:/VACSYN")
    gt_path = base_dir / "student_resource/dataset/train/train_ground_truth.tsv"
    s1_path = base_dir / "student_resource/dataset/train/train_source1.tsv"
    s2_path = base_dir / "student_resource/dataset/train/train_source2.tsv"
    s3_path = base_dir / "student_resource/dataset/train/train_source3.tsv"

    print("Sampling 500 S1 entities with ground truth...")
    s1_to_matches = defaultdict(list)
    needed_s2 = set()
    needed_s3 = set()
    needed_s1 = set()

    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1_id, _, rest = line.rstrip("\r\n").partition("\t")
            if rest:
                mids = [m.strip() for m in rest.split(",") if m.strip()]
                if mids:
                    s1_to_matches[s1_id] = mids
                    needed_s1.add(s1_id)
                    for m in mids:
                        if m.startswith("S2-"):
                            needed_s2.add(m)
                        elif m.startswith("S3-"):
                            needed_s3.add(m)
                    if len(needed_s1) >= 500:
                        break

    print(f"Sampled {len(needed_s1)} S1 entities.")
    print(f"Total true match pairs: {sum(len(m) for m in s1_to_matches.values())}")
    print(f"Needed S2 records: {len(needed_s2)}, S3 records: {len(needed_s3)}")

    # Load S1 records
    s1_rows = []
    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s1:
                s1_rows.append(parts)
                if len(s1_rows) >= len(needed_s1):
                    break
    s1_df = pd.DataFrame(s1_rows, columns=header)

    # Load S2 records (including 500 extra distractors for realism)
    s2_rows = []
    distractors_s2 = 0
    with open(s2_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s2:
                s2_rows.append(parts)
            elif distractors_s2 < 500:
                s2_rows.append(parts)
                distractors_s2 += 1
            if len(s2_rows) >= len(needed_s2) + 500:
                break
    s2_df = pd.DataFrame(s2_rows, columns=header)

    # Load S3 records (including 500 extra distractors)
    s3_rows = []
    distractors_s3 = 0
    with open(s3_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s3:
                s3_rows.append(parts)
            elif distractors_s3 < 500:
                s3_rows.append(parts)
                distractors_s3 += 1
            if len(s3_rows) >= len(needed_s3) + 500:
                break
    s3_df = pd.DataFrame(s3_rows, columns=header)

    print(f"Loaded: S1={len(s1_df)}, S2={len(s2_df)}, S3={len(s3_df)}")

    # Run CandidateGenerator
    start_t = time.perf_counter()
    cg = CandidateGenerator(max_candidates_per_s1=40)
    cand_df = cg.generate(s1_df, s2_df, s3_df)
    elapsed = time.perf_counter() - start_t

    print(f"Generated {len(cand_df)} candidate pairs in {elapsed:.2f}s")
    print(f"Columns: {list(cand_df.columns)}")

    # Evaluate Recall
    found_map = defaultdict(set)
    for _, row in cand_df.iterrows():
        found_map[row["s1_entity_id"]].add(row["candidate_entity_id"])

    total_true_pairs = 0
    recalled_pairs = 0
    s1_with_full_recall = 0

    for s1_id, true_mids in s1_to_matches.items():
        if s1_id not in s1_df["entity_id"].values:
            continue
        cands = found_map.get(s1_id, set())
        s1_recalled = 0
        for m in true_mids:
            total_true_pairs += 1
            if m in cands:
                recalled_pairs += 1
                s1_recalled += 1
        if s1_recalled == len(true_mids):
            s1_with_full_recall += 1

    recall = (recalled_pairs / total_true_pairs) * 100.0 if total_true_pairs > 0 else 0.0
    avg_cands_per_s1 = len(cand_df) / len(s1_df) if len(s1_df) > 0 else 0.0

    print("================ RESULTS ================")
    print(f"Total True Pairs: {total_true_pairs}")
    print(f"Recalled Pairs:   {recalled_pairs}")
    print(f"Candidate Recall: {recall:.2f}%")
    print(f"Average Candidates per S1: {avg_cands_per_s1:.2f}")
    print(f"Full S1 Entity Recall: {s1_with_full_recall} / {len(s1_df)} ({s1_with_full_recall/len(s1_df)*100:.1f}%)")
    print("=========================================")


if __name__ == "__main__":
    evaluate()
