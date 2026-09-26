"""
Stage 9 Automated Validation Test: Candidate Recall and Blocking Efficiency.

Verifies:
1. Candidate Recall >= 97.5% on representative ground truth training sample.
2. Candidate Efficiency: Average candidates per S1 entity <= 35.
3. Contract 1 Conformance: Output schema ['s1_entity_id', 'candidate_entity_id', 'candidate_source'].
"""

from collections import defaultdict
from pathlib import Path
import pandas as pd
import pytest

from business_entity_resolution.blocking.candidate_generator import CandidateGenerator


def test_candidate_recall_and_blocking_efficiency():
    """
    Validate that CandidateGenerator achieves >= 97.5% candidate recall
    while maintaining a tight candidate budget (<= 35 candidates/query).
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    gt_path = base_dir / "student_resource/dataset/train/train_ground_truth.tsv"
    s1_path = base_dir / "student_resource/dataset/train/train_source1.tsv"
    s2_path = base_dir / "student_resource/dataset/train/train_source2.tsv"
    s3_path = base_dir / "student_resource/dataset/train/train_source3.tsv"

    if not gt_path.exists() or not s1_path.exists() or not s2_path.exists() or not s3_path.exists():
        pytest.skip("Dataset files not found locally, skipping validation.")

    # Sample 200 S1 entities with ground truth matches
    needed_s1 = set()
    needed_s2 = set()
    needed_s3 = set()
    s1_to_matches = defaultdict(list)

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
                    if len(needed_s1) >= 200:
                        break

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

    # Load S2 records with 200 distractors
    s2_rows = []
    distractors_s2 = 0
    with open(s2_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s2:
                s2_rows.append(parts)
            elif distractors_s2 < 200:
                s2_rows.append(parts)
                distractors_s2 += 1
            if len(s2_rows) >= len(needed_s2) + 200:
                break
    s2_df = pd.DataFrame(s2_rows, columns=header)

    # Load S3 records with 200 distractors
    s3_rows = []
    distractors_s3 = 0
    with open(s3_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s3:
                s3_rows.append(parts)
            elif distractors_s3 < 200:
                s3_rows.append(parts)
                distractors_s3 += 1
            if len(s3_rows) >= len(needed_s3) + 200:
                break
    s3_df = pd.DataFrame(s3_rows, columns=header)

    # Execute CandidateGenerator
    cg = CandidateGenerator(max_candidates_per_s1=40)
    cand_df = cg.generate(s1_df, s2_df, s3_df)

    # 1. Verify Contract 1 schema
    expected_cols = ["s1_entity_id", "candidate_entity_id", "candidate_source"]
    assert list(cand_df.columns) == expected_cols, f"Columns must match Contract 1: {expected_cols}"
    assert not cand_df.duplicated(subset=["s1_entity_id", "candidate_entity_id"]).any(), "Candidate pairs must be unique"

    # 2. Verify Candidate Recall
    found_map = defaultdict(set)
    for _, row in cand_df.iterrows():
        found_map[row["s1_entity_id"]].add(row["candidate_entity_id"])

    total_true = 0
    recalled = 0
    for s1_id, true_mids in s1_to_matches.items():
        if s1_id not in s1_df["entity_id"].values:
            continue
        cands = found_map.get(s1_id, set())
        for m in true_mids:
            total_true += 1
            if m in cands:
                recalled += 1

    recall = recalled / total_true if total_true > 0 else 0.0
    avg_cands_per_s1 = len(cand_df) / len(s1_df) if len(s1_df) > 0 else 0.0

    print(f"\nStage 9 Validation: Recall = {recall*100:.2f}%, Avg Candidates/S1 = {avg_cands_per_s1:.2f}")

    # Assert recall target >= 97.5%
    assert recall >= 0.975, f"Candidate recall {recall*100:.2f}% is below target 97.5%"

    # Assert blocking efficiency <= 35.0 candidates per S1
    assert avg_cands_per_s1 <= 35.0, f"Average candidates {avg_cands_per_s1:.2f} exceeds budget 35.0"
