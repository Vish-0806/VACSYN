"""
Integration and End-to-End Pipeline Layer.

Responsible for orchestrating:
1. Data loading and memory-safe country-partitioned S1 chunking.
2. Candidate generation via frozen M1 CandidateGenerator.
3. Candidate submission formatting (source1_entity_id -> candidate_entity_ids).
4. Matching results formatting (source1_entity_id -> matched_entity_ids).
5. Candidate/Match consistency validation (matched_entity_ids ⊆ candidate_entity_ids).
6. Singleton preservation (guaranteeing 100% of test S1 reference entities appear).
7. Automated invocation of the official challenge submission validator.

Ownership: Member 4 (Integration & Submission).
"""

from collections import defaultdict
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import pandas as pd

from business_entity_resolution.blocking.candidate_generator import CandidateGenerator

logger = logging.getLogger(__name__)

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def format_candidate_pairs_submission(
    candidate_pairs_df: pd.DataFrame,
    all_s1_ids: Iterable[str],
) -> pd.DataFrame:
    """
    Convert pairwise M1 candidate pairs into the official competition submission format.

    Schema:
        source1_entity_id: Identifier of reference Source 1 entity.
        candidate_entity_ids: Comma-separated list of candidate IDs (or empty string for singletons).

    Guarantees:
        - Exactly one row per reference S1 entity in all_s1_ids.
        - Preserves all S1 entities (including zero-candidate singletons).
        - Comma-separated candidate IDs with internal duplicates removed.
        - No duplicate source1_entity_id rows.

    Args:
        candidate_pairs_df: DataFrame with ['s1_entity_id', 'candidate_entity_id', ...].
        all_s1_ids: Complete iterable of required Source 1 entity IDs.

    Returns:
        DataFrame conforming to the official candidate_pairs.tsv schema.
    """
    s1_to_cands: Dict[str, List[str]] = defaultdict(list)

    if candidate_pairs_df is not None and not candidate_pairs_df.empty:
        # Standardize column names if needed
        s1_col = "s1_entity_id" if "s1_entity_id" in candidate_pairs_df.columns else "source1_entity_id"
        cand_col = "candidate_entity_id" if "candidate_entity_id" in candidate_pairs_df.columns else "candidate_id"

        for _, row in candidate_pairs_df.iterrows():
            s1_id = str(row[s1_col]).strip()
            cid = str(row[cand_col]).strip()
            if s1_id and cid:
                s1_to_cands[s1_id].append(cid)

    # Reindex over all required S1 IDs preserving complete coverage
    rows: List[Dict[str, str]] = []
    seen_s1: Set[str] = set()

    for s1_id in all_s1_ids:
        s1_str = str(s1_id).strip()
        if not s1_str or s1_str in seen_s1:
            continue
        seen_s1.add(s1_str)

        raw_cands = s1_to_cands.get(s1_str, [])
        # Deduplicate while preserving insertion order
        unique_cands = list(dict.fromkeys(raw_cands))
        cand_str = ",".join(unique_cands) if unique_cands else ""

        rows.append({
            "source1_entity_id": s1_str,
            "candidate_entity_ids": cand_str,
        })

    return pd.DataFrame(rows, columns=CANDIDATE_HEADER)


def format_matching_results(
    predictions: Union[pd.DataFrame, Dict[str, Union[List[str], Set[str], str]]],
    all_s1_ids: Iterable[str],
) -> pd.DataFrame:
    """
    Format entity resolution predictions into the official matching_results.tsv schema.

    Schema:
        source1_entity_id: Identifier of reference Source 1 entity.
        matched_entity_ids: Comma-separated list of matched S2/S3 entity IDs,
                            or empty string for singletons (no matches).

    Guarantees:
        - Exactly one row per reference S1 entity in all_s1_ids.
        - Preserves singletons (unmatched S1 entities get an empty string).
        - Comma-separated match IDs with internal duplicates removed.
        - No duplicate source1_entity_id rows.

    Args:
        predictions: Mapping or DataFrame representing predicted matches.
                     If DataFrame, accepts ['source1_entity_id', 'matched_entity_ids']
                     or pairwise ['s1_entity_id', 'candidate_entity_id'].
                     If Dict, maps s1_entity_id to list/set of candidate IDs or comma-delimited string.
        all_s1_ids: Complete iterable of required Source 1 entity IDs.

    Returns:
        DataFrame conforming to the official matching_results.tsv schema.
    """
    s1_to_matches: Dict[str, List[str]] = defaultdict(list)

    if isinstance(predictions, pd.DataFrame) and not predictions.empty:
        cols_lower = {col.lower(): col for col in predictions.columns}
        if "source1_entity_id" in cols_lower and "matched_entity_ids" in cols_lower:
            s1_col = cols_lower["source1_entity_id"]
            match_col = cols_lower["matched_entity_ids"]
            for _, row in predictions.iterrows():
                s1_id = str(row[s1_col]).strip()
                val = str(row[match_col]).strip()
                if val:
                    mids = [m.strip() for m in val.split(",") if m.strip()]
                    s1_to_matches[s1_id].extend(mids)
        elif "s1_entity_id" in cols_lower and "candidate_entity_id" in cols_lower:
            s1_col = cols_lower["s1_entity_id"]
            cand_col = cols_lower["candidate_entity_id"]
            for _, row in predictions.iterrows():
                s1_id = str(row[s1_col]).strip()
                cid = str(row[cand_col]).strip()
                if s1_id and cid:
                    s1_to_matches[s1_id].append(cid)
    elif isinstance(predictions, dict):
        for s1_id, val in predictions.items():
            s1_str = str(s1_id).strip()
            if isinstance(val, str):
                mids = [m.strip() for m in val.split(",") if m.strip()]
                s1_to_matches[s1_str].extend(mids)
            elif isinstance(val, (list, set, tuple)):
                s1_to_matches[s1_str].extend([str(m).strip() for m in val if str(m).strip()])

    rows: List[Dict[str, str]] = []
    seen_s1: Set[str] = set()

    for s1_id in all_s1_ids:
        s1_str = str(s1_id).strip()
        if not s1_str or s1_str in seen_s1:
            continue
        seen_s1.add(s1_str)

        raw_matches = s1_to_matches.get(s1_str, [])
        unique_matches = list(dict.fromkeys(raw_matches))
        match_str = ",".join(unique_matches) if unique_matches else ""

        rows.append({
            "source1_entity_id": s1_str,
            "matched_entity_ids": match_str,
        })

    return pd.DataFrame(rows, columns=MATCHING_HEADER)


def validate_match_candidate_consistency(
    matching_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> bool:
    """
    Validate that every predicted match in matching_df is a subset of candidates in candidate_df.

    Rule:
        matched_entity_ids ⊆ candidate_entity_ids for every Source 1 entity.

    Raises:
        ValueError: If any predicted entity ID is not present in that S1's candidate set.

    Returns:
        True if all predicted matches are strictly consistent with candidate sets.
    """
    s1_col_cand = "source1_entity_id" if "source1_entity_id" in candidate_df.columns else "s1_entity_id"
    cand_col = "candidate_entity_ids" if "candidate_entity_ids" in candidate_df.columns else "candidate_entity_id"

    # Build S1 -> set(candidates) map
    s1_candidates: Dict[str, Set[str]] = defaultdict(set)
    for _, row in candidate_df.iterrows():
        s1_id = str(row[s1_col_cand]).strip()
        raw_val = str(row[cand_col]).strip()
        if raw_val:
            cids = [c.strip() for c in raw_val.split(",") if c.strip()]
            s1_candidates[s1_id].update(cids)

    s1_col_match = "source1_entity_id" if "source1_entity_id" in matching_df.columns else "s1_entity_id"
    match_col = "matched_entity_ids" if "matched_entity_ids" in matching_df.columns else "candidate_entity_id"

    for _, row in matching_df.iterrows():
        s1_id = str(row[s1_col_match]).strip()
        raw_match = str(row[match_col]).strip()
        if not raw_match:
            continue

        pred_ids = [p.strip() for p in raw_match.split(",") if p.strip()]
        valid_cands = s1_candidates.get(s1_id, set())

        for pred_id in pred_ids:
            if pred_id not in valid_cands:
                raise ValueError(
                    f"Consistency violation: Predicted match '{pred_id}' for S1 '{s1_id}' "
                    f"is not in that S1's candidate set. Available candidates: {sorted(valid_cands)}"
                )

    return True


def generate_candidates_chunked(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    chunk_size: int = 100_000,
    max_candidates_per_s1: int = 50,
) -> pd.DataFrame:
    """
    Execute country-partitioned, chunked candidate generation using frozen M1 CandidateGenerator.

    Maintains low memory footprint:
    1. Partitions candidate pool (S2 + S3) by country.
    2. Partitions reference S1 queries by country.
    3. Slices S1 in chunks of `chunk_size` within each country partition.
    4. Executes CandidateGenerator.generate on each slice against the country candidate pool.
    5. Deduplicates and unions results.

    Args:
        s1_df: Reference S1 DataFrame [entity_id, business_name, business_address, country].
        s2_df: Candidate pool S2 DataFrame.
        s3_df: Candidate pool S3 DataFrame.
        chunk_size: Number of S1 queries per chunk.
        max_candidates_per_s1: Candidate budget per S1 entity.

    Returns:
        Pairwise candidate DataFrame with columns:
        ['s1_entity_id', 'candidate_entity_id', 'candidate_source']
    """
    if s1_df is None or s1_df.empty:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source"])

    s1_clean = s1_df.copy()
    s1_clean["country_norm"] = s1_clean["country"].fillna("").astype(str).str.strip().str.upper()

    s2_clean = s2_df.copy() if s2_df is not None and not s2_df.empty else pd.DataFrame()
    if not s2_clean.empty:
        s2_clean["country_norm"] = s2_clean["country"].fillna("").astype(str).str.strip().str.upper()

    s3_clean = s3_df.copy() if s3_df is not None and not s3_df.empty else pd.DataFrame()
    if not s3_clean.empty:
        s3_clean["country_norm"] = s3_clean["country"].fillna("").astype(str).str.strip().str.upper()

    countries = s1_clean["country_norm"].unique()
    all_chunk_pairs: List[pd.DataFrame] = []

    generator = CandidateGenerator(max_candidates_per_s1=max_candidates_per_s1)

    for country in countries:
        if not country:
            continue

        s1_country = s1_clean[s1_clean["country_norm"] == country]
        s2_country = s2_clean[s2_clean["country_norm"] == country] if not s2_clean.empty else pd.DataFrame()
        s3_country = s3_clean[s3_clean["country_norm"] == country] if not s3_clean.empty else pd.DataFrame()

        if s1_country.empty or (s2_country.empty and s3_country.empty):
            continue

        n_s1 = len(s1_country)
        # Process S1 in chunks
        for start_idx in range(0, n_s1, chunk_size):
            end_idx = min(start_idx + chunk_size, n_s1)
            s1_slice = s1_country.iloc[start_idx:end_idx]

            chunk_pairs = generator.generate(
                s1_df=s1_slice,
                s2_df=s2_country,
                s3_df=s3_country,
            )
            if chunk_pairs is not None and not chunk_pairs.empty:
                all_chunk_pairs.append(chunk_pairs)

    if not all_chunk_pairs:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source"])

    combined_df = pd.concat(all_chunk_pairs, ignore_index=True)
    combined_df.drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"], inplace=True)
    return combined_df


def run_submission_validator(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
    check_ids: bool = False,
) -> int:
    """
    Invoke the official submission validator (student_resource/utils/validate_submission.py).

    Args:
        matching_path: Path to generated matching_results.tsv.
        candidate_path: Path to generated candidate_pairs.tsv.
        test_dir: Directory containing test source files (test_source1.tsv, etc.).
        check_ids: If True, check that matched/candidate IDs exist in test source files.

    Returns:
        Exit code from validate_submission.py (0 = passed, 1 = failed).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    possible_paths = [
        repo_root / "student_resource" / "utils" / "validate_submission.py",
        repo_root / "utils" / "validate_submission.py",
    ]

    validator_path = None
    for p in possible_paths:
        if p.exists():
            validator_path = p
            break

    if validator_path is None:
        raise FileNotFoundError(
            f"Official submission validator not found in: {[str(p) for p in possible_paths]}"
        )

    cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]
    if check_ids:
        cmd.append("--check-ids")

    logger.info("Executing official submission validator: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)

    return res.returncode


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    model_path: Optional[Path] = None,
    is_train: bool = False,
    chunk_size: int = 100_000,
    max_candidates_per_s1: int = 50,
    validate: bool = False,
) -> None:
    """
    Execute the M4 integration and submission generation pipeline.

    Workflow:
        1. Resolve source data files in data_dir (train or test).
        2. Read all Source 1 entity IDs to guarantee 100% coverage.
        3. Execute memory-safe country-partitioned S1 chunking via CandidateGenerator.
        4. Format and write official candidate_pairs.tsv.
        5. When ML model is available, score pairs and apply thresholding (M2).
        6. Format and write official matching_results.tsv.
        7. Validate consistency (matched ⊆ candidates).
        8. Optionally execute official challenge submission validator.

    Args:
        data_dir: Path to directory containing source TSVs (train or test).
        output_dir: Path to directory where submission files will be written.
        model_path: Optional path to pre-trained LightGBM model artifact.
        is_train: Flag indicating whether running on train (with ground truth) or test.
        chunk_size: Batch size for chunked streaming to maintain low memory usage.
        max_candidates_per_s1: Maximum candidate budget per S1 entity.
        validate: Whether to run the official submission validator after generation.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "train" if is_train else "test"
    s1_file = data_dir / f"{prefix}_source1.tsv"
    s2_file = data_dir / f"{prefix}_source2.tsv"
    s3_file = data_dir / f"{prefix}_source3.tsv"

    for fpath in [s1_file, s2_file, s3_file]:
        if not fpath.exists():
            raise FileNotFoundError(f"Required source file not found: {fpath}")

    # 1. Read all S1 entity IDs to guarantee singleton preservation
    all_s1_ids: List[str] = []
    with open(s1_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.split(DELIM, 1)
            if parts and parts[0].strip():
                all_s1_ids.append(parts[0].strip())

    logger.info("Loaded %d reference Source 1 entity IDs from %s", len(all_s1_ids), s1_file)

    # 2. Load sources and execute chunked candidate generation
    s1_df = pd.read_csv(s1_file, sep=DELIM, dtype=str)
    s2_df = pd.read_csv(s2_file, sep=DELIM, dtype=str)
    s3_df = pd.read_csv(s3_file, sep=DELIM, dtype=str)

    logger.info("Executing country-partitioned S1 chunked blocking (chunk_size=%d)...", chunk_size)
    candidate_pairs_df = generate_candidates_chunked(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        chunk_size=chunk_size,
        max_candidates_per_s1=max_candidates_per_s1,
    )

    # 3. Format and write official candidate_pairs.tsv
    cand_submission_df = format_candidate_pairs_submission(
        candidate_pairs_df=candidate_pairs_df,
        all_s1_ids=all_s1_ids,
    )
    cand_out_path = output_dir / "candidate_pairs.tsv"
    cand_submission_df.to_csv(cand_out_path, sep=DELIM, index=False)
    logger.info("Saved official candidate pairs to %s (%d rows)", cand_out_path, len(cand_submission_df))

    # 4. Check model availability for ML matching
    if model_path is not None:
        raise NotImplementedError(
            "M2 model prediction and feature extraction interfaces are not yet integrated. "
            "Model scoring is blocked until Member 2 finishes implementation."
        )

    # 5. Optional validation check if matching results already exists
    matching_out_path = output_dir / "matching_results.tsv"
    if matching_out_path.exists():
        matching_df = pd.read_csv(matching_out_path, sep=DELIM, dtype=str)
        validate_match_candidate_consistency(matching_df, cand_submission_df)

        if validate:
            exit_code = run_submission_validator(
                matching_path=matching_out_path,
                candidate_path=cand_out_path,
                test_dir=data_dir,
            )
            if exit_code != 0:
                raise ValueError("Official submission validator reported formatting errors.")
