"""
Distributed Training-Data Shard Processor.

Responsible for:
1. Deterministic S1 sharding (50/50 split via MD5 hash on S1 entity ID).
2. S1 streaming / loading for the target shard (with optional pilot limit).
3. Full S2/S3 candidate pool loading.
4. Candidate generation via frozen M1 CandidateGenerator.
5. Pairwise feature extraction via existing M2 PairFeatureExtractor.
6. Ground-truth binary labeling (1 = true match, 0 = non-match).
7. Validation checks (schema, duplicates, missing values, recall).
8. Persisting training shard artifact (Parquet / TSV).

Ownership: M1 (Chinmay) - Distributed training-data orchestration.
Frozen dependencies: M1 Preprocessing & Blocking, M2 Feature Extraction.
"""

import argparse
import hashlib
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / "student_resource").exists() and (p / "business-entity-resolution").exists():
            return p
    return current.parent.parent.parent.parent.parent

_REPO_ROOT = find_repo_root()
_SRC_DIR = _REPO_ROOT / "business-entity-resolution" / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from business_entity_resolution.blocking.candidate_generator import CandidateGenerator
from business_entity_resolution.features.pair_features import PairFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("process_training_shard")

EXPECTED_FEATURE_COLUMNS = [
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_char_ngram_cosine",
    "name_edit_similarity",
    "name_length_ratio",
    "name_exact_match",
    "name_first_token_match",
    "address_token_jaccard",
    "address_token_overlap_count",
    "address_house_number_match",
    "address_edit_similarity",
    "address_length_ratio",
    "address_missing",
    "country_match",
    "source_is_s2",
    "source_is_s3",
    "name_length_ratio_context",
    "address_length_ratio_context",
]

FINAL_SCHEMA_COLUMNS = [
    "s1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    *EXPECTED_FEATURE_COLUMNS,
    "label",
]


def get_shard(s1_entity_id: str) -> int:
    """
    Deterministic 50/50 partition based strictly on S1 entity ID MD5 hash.

    Guarantee:
        - Every S1 entity belongs to exactly one shard (0 or 1).
        - No S1 entity belongs to both shards.
        - Independent of DataFrame row order or machine environment.

    Args:
        s1_entity_id: Source 1 entity identifier (e.g. 'S1-925783039').

    Returns:
        0 (Chinmay / M1) or 1 (Vishal / M2).
    """
    return int(
        hashlib.md5(str(s1_entity_id).encode("utf-8")).hexdigest(),
        16,
    ) % 2


def resolve_data_paths(data_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Locate training dataset TSV files in known project directories.
    """
    possible_roots = []
    if data_dir:
        possible_roots.append(Path(data_dir))
    possible_roots.extend([
        _REPO_ROOT / "student_resource" / "dataset" / "train",
        _REPO_ROOT / "business-entity-resolution" / "dataset" / "train",
        _REPO_ROOT / "dataset" / "train",
    ])

    for root in possible_roots:
        s1 = root / "train_source1.tsv"
        s2 = root / "train_source2.tsv"
        s3 = root / "train_source3.tsv"
        gt = root / "train_ground_truth.tsv"
        if s1.exists() and s2.exists() and s3.exists() and gt.exists():
            logger.info("Found training dataset at: %s", root)
            return {
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "gt": gt,
            }

    raise FileNotFoundError(
        f"Could not locate training dataset TSV files. Searched paths: {[str(p) for p in possible_roots]}"
    )


def load_s1_shard(
    s1_path: Path,
    target_shard: int,
    limit_s1: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stream and filter S1 records belonging strictly to target_shard.

    Args:
        s1_path: Path to train_source1.tsv.
        target_shard: Shard index (0 or 1).
        limit_s1: Maximum number of S1 entities to load (for pilot runs).

    Returns:
        DataFrame with columns ['entity_id', 'business_name', 'business_address', 'country'].
    """
    logger.info("Streaming S1 records for shard %d (limit=%s)...", target_shard, limit_s1)
    rows: List[List[str]] = []
    with open(s1_path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        header = [h.strip() for h in header_line.split("\t")]
        id_idx = header.index("entity_id") if "entity_id" in header else 0

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or len(parts) <= id_idx:
                continue
            s1_id = parts[id_idx].strip()
            if get_shard(s1_id) == target_shard:
                rows.append(parts)
                if limit_s1 is not None and len(rows) >= limit_s1:
                    break

    df = pd.DataFrame(rows, columns=header)
    logger.info("Loaded %d S1 records for shard %d", len(df), target_shard)
    return df


def load_candidate_pool(source_path: Path, source_name: str) -> pd.DataFrame:
    """
    Load a full candidate source DataFrame (S2 or S3).
    """
    logger.info("Loading full %s candidate pool from %s...", source_name, source_path)
    t0 = time.perf_counter()
    cols = ["entity_id", "business_name", "business_address", "country"]
    df = pd.read_csv(source_path, sep="\t", dtype=str, usecols=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["source"] = source_name
    elapsed = time.perf_counter() - t0
    logger.info("Loaded %s in %.2fs (%d rows)", source_name, elapsed, len(df))
    return df


def load_ground_truth(
    gt_path: Path,
    target_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Stream ground truth matches, optionally filtered to target S1 entities.

    Format:
        source1_entity_id -> set(matched_candidate_entity_ids)
    """
    logger.info("Loading ground truth from %s...", gt_path)
    gt_map: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue
            s1_id, _, rest = line_str.partition("\t")
            s1_id = s1_id.strip()
            if target_s1_ids is not None and s1_id not in target_s1_ids:
                continue
            matches = set()
            if rest.strip():
                for m in rest.split(","):
                    m_clean = m.strip()
                    if m_clean:
                        matches.add(m_clean)
            gt_map[s1_id] = matches

    logger.info("Loaded ground truth for %d S1 entities", len(gt_map))
    return gt_map


def build_candidate_record_lookups(
    candidate_pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Construct record lookup mappings for PairFeatureExtractor.

    To conserve memory and CPU, builds candidate lookups ONLY for candidate
    entities that appear in the candidate_pairs_df.
    """
    logger.info("Building entity record lookups for feature extraction...")
    t0 = time.perf_counter()

    # S1 lookup: indexed by entity_id
    s1_records: Dict[str, Dict[str, Any]] = s1_df.set_index("entity_id", drop=False)[
        ["entity_id", "business_name", "business_address", "country"]
    ].to_dict(orient="index")

    # Determine unique S2 and S3 candidate IDs needed
    needed_s2 = set(
        candidate_pairs_df[candidate_pairs_df["candidate_source"] == "S2"]["candidate_entity_id"]
    )
    needed_s3 = set(
        candidate_pairs_df[candidate_pairs_df["candidate_source"] == "S3"]["candidate_entity_id"]
    )

    logger.info("Required candidate records: %d from S2, %d from S3", len(needed_s2), len(needed_s3))

    s2_subset = s2_df[s2_df["entity_id"].isin(needed_s2)]
    s2_records: Dict[str, Dict[str, Any]] = s2_subset.set_index("entity_id", drop=False)[
        ["entity_id", "business_name", "business_address", "country"]
    ].to_dict(orient="index")

    s3_subset = s3_df[s3_df["entity_id"].isin(needed_s3)]
    s3_records: Dict[str, Dict[str, Any]] = s3_subset.set_index("entity_id", drop=False)[
        ["entity_id", "business_name", "business_address", "country"]
    ].to_dict(orient="index")

    elapsed = time.perf_counter() - t0
    logger.info("Record lookups assembled in %.2fs", elapsed)
    return s1_records, s2_records, s3_records


def validate_training_shard(
    df: pd.DataFrame,
    target_shard: int,
    ground_truth: Dict[str, Set[str]],
    processed_s1_ids: Set[str],
) -> Dict[str, Any]:
    """
    Execute mandatory validation checks on the processed training shard.
    """
    logger.info("Executing validation checks on training shard...")
    stats: Dict[str, Any] = {}

    # Check 1: Shard correctness
    invalid_shard_ids = [
        sid for sid in df["s1_entity_id"].unique() if get_shard(sid) != target_shard
    ]
    if invalid_shard_ids:
        raise ValueError(
            f"Shard validation failed: {len(invalid_shard_ids)} S1 entities belong to wrong shard! "
            f"Examples: {invalid_shard_ids[:5]}"
        )
    stats["shard_correctness"] = True

    # Check 2: No duplicate candidate pairs
    duplicate_mask = df.duplicated(subset=["s1_entity_id", "candidate_entity_id"])
    if duplicate_mask.any():
        num_dups = duplicate_mask.sum()
        raise ValueError(f"Duplicate candidate pairs detected: {num_dups} duplicate rows.")
    stats["no_duplicates"] = True

    # Check 3: Candidate source validity
    invalid_sources = set(df["candidate_source"].unique()) - {"S2", "S3"}
    if invalid_sources:
        raise ValueError(f"Invalid candidate source detected: {invalid_sources}. Expected 'S2' or 'S3'.")
    stats["candidate_sources"] = list(df["candidate_source"].value_counts().to_dict().items())

    # Check 4: No missing labels
    if df["label"].isna().any():
        raise ValueError("Missing label detected in candidate pairs.")
    invalid_labels = set(df["label"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Invalid label values detected: {invalid_labels}. Expected only 0 or 1.")
    stats["no_missing_labels"] = True

    # Check 5: Feature completeness (all 18 features, no NaN / Inf)
    for col in EXPECTED_FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")
        if df[col].isna().any():
            raise ValueError(f"NaN detected in feature column: {col}")
        if np.isinf(df[col]).any():
            raise ValueError(f"Inf detected in feature column: {col}")
    stats["features_complete"] = True

    # Check 6: Label statistics
    num_pos = int((df["label"] == 1).sum())
    num_neg = int((df["label"] == 0).sum())
    total_candidates = len(df)
    pos_rate = (num_pos / total_candidates * 100.0) if total_candidates > 0 else 0.0
    stats["positive_labels"] = num_pos
    stats["negative_labels"] = num_neg
    stats["positive_rate_pct"] = pos_rate

    # Check 7: Candidate count distribution
    s1_counts = df["s1_entity_id"].value_counts()
    num_s1 = len(processed_s1_ids)
    s1_with_cands = len(s1_counts)
    s1_zero_cands = num_s1 - s1_with_cands

    avg_cands = total_candidates / num_s1 if num_s1 > 0 else 0.0
    median_cands = float(s1_counts.median()) if not s1_counts.empty else 0.0
    max_cands = int(s1_counts.max()) if not s1_counts.empty else 0
    num_at_cap = int((s1_counts >= max_cands).sum()) if not s1_counts.empty else 0

    stats["s1_total_count"] = num_s1
    stats["s1_with_candidates"] = s1_with_cands
    stats["s1_zero_candidates"] = s1_zero_cands
    stats["candidate_total_count"] = total_candidates
    stats["avg_candidates_per_s1"] = avg_cands
    stats["median_candidates_per_s1"] = median_cands
    stats["max_candidates_per_s1"] = max_cands
    stats["num_s1_at_candidate_cap"] = num_at_cap

    # Check 8: Ground-truth coverage (candidate recall)
    total_true_pairs = 0
    recalled_pairs = 0
    for s1_id in processed_s1_ids:
        true_mids = ground_truth.get(s1_id, set())
        total_true_pairs += len(true_mids)

    recalled_pairs = num_pos
    recall_pct = (recalled_pairs / total_true_pairs * 100.0) if total_true_pairs > 0 else 0.0
    stats["total_ground_truth_pairs"] = total_true_pairs
    stats["recalled_ground_truth_pairs"] = recalled_pairs
    stats["candidate_recall_pct"] = recall_pct

    return stats


def get_memory_usage_mb() -> float:
    """Return process resident set size in megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def process_training_shard(
    shard: int = 0,
    limit_s1: Optional[int] = None,
    chunk_size: int = 1000,
    output_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    output_format: str = "parquet",
    max_candidates_per_s1: int = 100,
    max_bucket_size: int = 100,
    max_token_doc_freq: int = 2000,
    tfidf_top_k: int = 40,
    resume: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute chunked training-data shard generation for the specified shard.

    Workflow per chunk:
        Selected Shard 0 S1
                ↓
        split into chunks of chunk_size (default 1,000 S1)
                ↓
        CandidateGenerator on chunk
                ↓
        M2 18-feature extraction
                ↓
        ground-truth labeling
                ↓
        validate
                ↓
        persist chunk result
                ↓
        next chunk

    Args:
        shard: Deterministic shard index (0 = Chinmay, 1 = Vishal).
        limit_s1: Maximum S1 entities to process (for pilot evaluation).
        chunk_size: S1 batch size per chunk (default 1,000 to prevent OOM in TF-IDF).
        output_path: Destination path or directory for training shard artifacts.
        data_dir: Root directory of training dataset TSVs.
        output_format: 'parquet' (preferred) or 'tsv'.
        max_candidates_per_s1: Candidate budget per S1 entity (frozen at 50).
        max_bucket_size: Maximum bucket size for exact/prefix index.
        max_token_doc_freq: Maximum document frequency for address/name tokens.
        resume: Whether to detect and skip already completed chunks.

    Returns:
        Tuple of (list of chunk stats dictionaries, total summary dictionary).
    """
    total_start = time.perf_counter()
    logger.info("================================================================")
    logger.info("STARTING CHUNKED TRAINING SHARD GENERATION - SHARD %d", shard)
    logger.info(
        "Parameters: limit_s1=%s, chunk_size=%d, max_cands=%d, max_bucket=%d, max_freq=%d, format=%s",
        limit_s1, chunk_size, max_candidates_per_s1, max_bucket_size, max_token_doc_freq, output_format
    )
    logger.info("Initial Memory: %.1f MB", get_memory_usage_mb())
    logger.info("================================================================")

    # 1. Resolve dataset paths
    paths = resolve_data_paths(data_dir)

    # 2. Stream S1 for target shard
    t0 = time.perf_counter()
    s1_df = load_s1_shard(paths["s1"], target_shard=shard, limit_s1=limit_s1)
    total_s1 = len(s1_df)
    processed_s1_ids = set(s1_df["entity_id"])
    logger.info("S1 Shard Loaded: %d records in %.2fs (Memory: %.1f MB)", total_s1, time.perf_counter() - t0, get_memory_usage_mb())

    # 3. Stream ground truth for target S1 entities
    gt_map = load_ground_truth(paths["gt"], target_s1_ids=processed_s1_ids)

    # 4. Load full S2 and S3 candidate pools (never sharded)
    s2_df = load_candidate_pool(paths["s2"], source_name="S2")
    s3_df = load_candidate_pool(paths["s3"], source_name="S3")
    logger.info("Full Candidate Pools Loaded (Memory: %.1f MB)", get_memory_usage_mb())

    # 5. Determine chunk directory and artifact naming
    if output_path is not None:
        p_out = Path(output_path)
        if p_out.suffix in (".parquet", ".tsv"):
            out_dir = p_out.parent
            custom_single_file = p_out
        else:
            out_dir = p_out
            custom_single_file = None
    else:
        out_dir = _REPO_ROOT / "business-entity-resolution" / "output" / "training_shards"
        custom_single_file = None

    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "parquet" if output_format == "parquet" else "tsv"

    # Compute chunk slices
    n_chunks = (total_s1 + chunk_size - 1) // chunk_size if total_s1 > 0 else 0
    logger.info("Total S1 records: %d | Chunk size: %d | Total chunks: %d", total_s1, chunk_size, n_chunks)

    generator = CandidateGenerator(
        max_candidates_per_s1=max_candidates_per_s1,
        max_bucket_size=max_bucket_size,
        max_token_doc_freq=max_token_doc_freq,
        tfidf_top_k=tfidf_top_k,
    )
    extractor = PairFeatureExtractor()
    chunk_reports: List[Dict[str, Any]] = []

    for chunk_idx in range(n_chunks):
        c_start_idx = chunk_idx * chunk_size
        c_end_idx = min(c_start_idx + chunk_size, total_s1)
        s1_chunk = s1_df.iloc[c_start_idx:c_end_idx].copy()
        chunk_s1_ids = set(s1_chunk["entity_id"])
        chunk_gt_map = {sid: gt_map.get(sid, set()) for sid in chunk_s1_ids}

        if custom_single_file is not None and n_chunks == 1:
            chunk_file_path = custom_single_file
        else:
            chunk_filename = f"candidate_features_labels_shard_{shard}_chunk_{chunk_idx:03d}.{ext}"
            chunk_file_path = out_dir / chunk_filename

        logger.info("-" * 60)
        logger.info("PROCESSING CHUNK %d / %d (S1 indices %d to %d, size=%d)", chunk_idx, n_chunks - 1, c_start_idx, c_end_idx - 1, len(s1_chunk))
        logger.info("Target chunk artifact: %s", chunk_file_path)

        # Resume check
        if resume and chunk_file_path.exists() and chunk_file_path.stat().st_size > 0:
            logger.info("Chunk %d already exists at %s. Loading existing artifact...", chunk_idx, chunk_file_path)
            if ext == "parquet":
                existing_df = pd.read_parquet(chunk_file_path)
            else:
                existing_df = pd.read_csv(chunk_file_path, sep="\t", dtype=str)

            chunk_stats = validate_training_shard(
                df=existing_df,
                target_shard=shard,
                ground_truth=chunk_gt_map,
                processed_s1_ids=chunk_s1_ids,
            )
            chunk_stats["chunk_idx"] = chunk_idx
            chunk_stats["time_candidates_s"] = 0.0
            chunk_stats["time_features_s"] = 0.0
            chunk_stats["time_labeling_s"] = 0.0
            chunk_stats["time_total_s"] = 0.0
            chunk_stats["output_file"] = str(chunk_file_path)
            chunk_stats["output_size_bytes"] = chunk_file_path.stat().st_size
            chunk_stats["resumed"] = True
            chunk_reports.append(chunk_stats)
            logger.info("Resumed chunk %d: %d candidates, %d positives", chunk_idx, len(existing_df), chunk_stats["positive_labels"])
            continue

        chunk_t0 = time.perf_counter()

        # Step 5a: Candidate generation for this chunk
        logger.info("Running CandidateGenerator on chunk %d (%d S1 records)...", chunk_idx, len(s1_chunk))
        t_cand = time.perf_counter()
        candidate_pairs_df = generator.generate(s1_df=s1_chunk, s2_df=s2_df, s3_df=s3_df)
        elapsed_cand = time.perf_counter() - t_cand
        logger.info("Chunk %d Candidate generation complete: %d pairs in %.2fs", chunk_idx, len(candidate_pairs_df), elapsed_cand)

        # Step 5b: Feature extraction for this chunk
        logger.info("Extracting 18 M2 features for chunk %d...", chunk_idx)
        t_feat = time.perf_counter()
        s1_rec, s2_rec, s3_rec = build_candidate_record_lookups(
            candidate_pairs_df=candidate_pairs_df,
            s1_df=s1_chunk,
            s2_df=s2_df,
            s3_df=s3_df,
        )
        features_df = extractor.extract_features(
            candidate_pairs_df=candidate_pairs_df,
            s1_records=s1_rec,
            s2_records=s2_rec,
            s3_records=s3_rec,
        )
        elapsed_feat = time.perf_counter() - t_feat
        logger.info("Chunk %d Feature extraction complete in %.2fs", chunk_idx, elapsed_feat)

        # Step 5c: Ground-truth labeling for this chunk
        logger.info("Assigning ground-truth binary labels for chunk %d...", chunk_idx)
        t_label = time.perf_counter()
        labels: List[int] = []
        for s1_id, cand_id in zip(features_df["s1_entity_id"], features_df["candidate_entity_id"]):
            true_mids = chunk_gt_map.get(str(s1_id).strip(), set())
            label = 1 if str(cand_id).strip() in true_mids else 0
            labels.append(label)
        features_df["label"] = pd.Series(labels, dtype=int)
        elapsed_label = time.perf_counter() - t_label

        final_chunk_df = features_df[FINAL_SCHEMA_COLUMNS].copy()

        # Step 5d: Validate chunk
        chunk_stats = validate_training_shard(
            df=final_chunk_df,
            target_shard=shard,
            ground_truth=chunk_gt_map,
            processed_s1_ids=chunk_s1_ids,
        )
        chunk_total_elapsed = time.perf_counter() - chunk_t0
        chunk_stats["chunk_idx"] = chunk_idx
        chunk_stats["time_candidates_s"] = elapsed_cand
        chunk_stats["time_features_s"] = elapsed_feat
        chunk_stats["time_labeling_s"] = elapsed_label
        chunk_stats["time_total_s"] = chunk_total_elapsed
        chunk_stats["resumed"] = False

        # Step 5e: Persist chunk artifact immediately
        logger.info("Persisting chunk %d artifact to %s...", chunk_idx, chunk_file_path)
        if ext == "parquet":
            final_chunk_df.to_parquet(chunk_file_path, index=False)
        else:
            final_chunk_df.to_csv(chunk_file_path, sep="\t", index=False)

        # Verify output is readable
        if ext == "parquet":
            _test_read = pd.read_parquet(chunk_file_path)
        else:
            _test_read = pd.read_csv(chunk_file_path, sep="\t", dtype=str)
        if len(_test_read) != len(final_chunk_df):
            raise IOError(f"Written chunk verification failed: expected {len(final_chunk_df)} rows, read {len(_test_read)}")

        chunk_size_bytes = chunk_file_path.stat().st_size
        chunk_stats["output_file"] = str(chunk_file_path)
        chunk_stats["output_size_bytes"] = chunk_size_bytes
        chunk_reports.append(chunk_stats)

        logger.info(
            "Chunk %d successfully completed in %.2fs (Size: %.2f MB, Memory: %.1f MB)",
            chunk_idx,
            chunk_total_elapsed,
            chunk_size_bytes / (1024 * 1024),
            get_memory_usage_mb(),
        )

    # 6. Overall summary
    total_elapsed = time.perf_counter() - total_start
    total_summary = {
        "shard": shard,
        "total_s1": total_s1,
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "time_total_s": total_elapsed,
        "peak_memory_mb": get_memory_usage_mb(),
        "chunks": chunk_reports,
    }

    _print_chunked_report(chunk_reports, total_summary)
    return chunk_reports, total_summary


def _print_chunked_report(chunk_reports: List[Dict[str, Any]], total_summary: Dict[str, Any]) -> None:
    """Print structured per-chunk and total execution report."""
    print("\n" + "=" * 65)
    print("        DISTRIBUTED TRAINING SHARD - CHUNK EXECUTION REPORT")
    print("=" * 65)

    for cr in chunk_reports:
        c_idx = cr.get("chunk_idx", 0)
        c_size_mb = cr.get("output_size_bytes", 0) / (1024 * 1024)
        total_t = cr.get('time_total_s', 0.0)
        cand_t = cr.get('time_candidates_s', 0.0)
        cand_pct = (cand_t / total_t * 100.0) if total_t > 0 else 0.0

        print(f"\nChunk {c_idx}:")
        print(f"  runtime:                    {total_t:.2f}s (cand: {cand_t:.2f}s [{cand_pct:.1f}%], feat: {cr.get('time_features_s', 0.0):.2f}s, label: {cr.get('time_labeling_s', 0.0):.2f}s)")
        print(f"  candidates:                 {cr.get('candidate_total_count', 0):,}")
        print(f"  avg candidates/S1:          {cr.get('avg_candidates_per_s1', 0.0):.2f}")
        print(f"  max candidates/S1:          {cr.get('max_candidates_per_s1', 0):,}")
        print(f"  S1 at candidate cap:        {cr.get('num_s1_at_candidate_cap', 0):,}")
        print(f"  positives:                  {cr.get('positive_labels', 0):,}")
        print(f"  negatives:                  {cr.get('negative_labels', 0):,}")
        print(f"  recall:                     {cr.get('candidate_recall_pct', 0.0):.2f}% ({cr.get('recalled_ground_truth_pairs', 0):,} / {cr.get('total_ground_truth_pairs', 0):,})")
        print(f"  output size:                {c_size_mb:.2f} MB ({cr.get('output_file', '')})")

    # Time projections
    valid_runtimes = [cr["time_total_s"] for cr in chunk_reports if not cr.get("resumed", False) and cr["time_total_s"] > 0]
    avg_chunk_time = (sum(valid_runtimes) / len(valid_runtimes)) if valid_runtimes else 0.0
    time_10k = avg_chunk_time * 10.0
    time_full_1_1m = avg_chunk_time * 1100.0

    print("\n" + "-" * 65)
    print("Total:")
    print(f"  runtime:                    {total_summary.get('time_total_s', 0.0):.2f}s (~{total_summary.get('time_total_s', 0.0) / 60.0:.1f} min)")
    print(f"  approximate memory:         {total_summary.get('peak_memory_mb', 0.0):.1f} MB")
    print(f"  estimated time for 10K:     {time_10k:.1f}s (~{time_10k / 60.0:.1f} min / ~{time_10k / 3600.0:.2f} hrs)")
    print(f"  estimated time for ~1.1M:   {time_full_1_1m:.1f}s (~{time_full_1_1m / 3600.0:.1f} hrs / ~{time_full_1_1m / 86400.0:.1f} days)")
    print(f"  S2/S3 re-indexing behavior: REBUILT FOR EVERY CHUNK (frozen CandidateGenerator rebuilds indexes per call)")
    print("=" * 65 + "\n")


def _print_report(stats: Dict[str, Any]) -> None:
    """Print clean execution and validation summary."""
    print("\n" + "=" * 60)
    print("           TRAINING SHARD EXECUTION REPORT")
    print("=" * 60)
    print(f"S1 Reference Entities:       {stats.get('s1_total_count', 0):,}")
    print(f"S1 with Candidates:          {stats.get('s1_with_candidates', 0):,}")
    print(f"S1 Zero Candidates:          {stats.get('s1_zero_candidates', 0):,}")
    print(f"Total Candidate Pairs:       {stats.get('candidate_total_count', 0):,}")
    print(f"Average Candidates / S1:     {stats.get('avg_candidates_per_s1', 0.0):.2f}")
    print(f"Median Candidates / S1:      {stats.get('median_candidates_per_s1', 0.0):.1f}")
    print(f"Max Candidates / S1:         {stats.get('max_candidates_per_s1', 0):,}")
    print(f"S1 at Candidate Cap:         {stats.get('num_s1_at_candidate_cap', 0):,}")
    print("-" * 60)
    print(f"Positive Labels (1):         {stats.get('positive_labels', 0):,}")
    print(f"Negative Labels (0):         {stats.get('negative_labels', 0):,}")
    print(f"Positive Rate:               {stats.get('positive_rate_pct', 0.0):.2f}%")
    print("-" * 60)
    print(f"Ground Truth True Pairs:     {stats.get('total_ground_truth_pairs', 0):,}")
    print(f"Recalled Ground Truth Pairs: {stats.get('recalled_ground_truth_pairs', 0):,}")
    print(f"Candidate Recall:            {stats.get('candidate_recall_pct', 0.0):.2f}%")
    print("-" * 60)
    print(f"Candidate Generation Time:   {stats.get('time_candidates_s', 0.0):.2f}s")
    print(f"Feature Extraction Time:     {stats.get('time_features_s', 0.0):.2f}s")
    print(f"Total Processing Time:       {stats.get('time_total_s', 0.0):.2f}s")
    print(f"Current Process Memory:      {stats.get('peak_memory_mb', 0.0):.1f} MB")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process and build distributed training data shard.")
    parser.add_argument("--shard", type=int, default=0, choices=[0, 1], help="Deterministic shard ID (0=Chinmay, 1=Vishal).")
    parser.add_argument("--limit-s1", type=int, default=None, help="Limit number of S1 entities processed (pilot mode).")
    parser.add_argument("--chunk-size", type=int, default=1000, help="S1 chunk size for batch processing (default: 1000).")
    parser.add_argument("--output", type=str, default=None, help="Custom output filepath for training shard.")
    parser.add_argument("--format", type=str, default="parquet", choices=["parquet", "tsv"], help="Output format.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to training dataset directory.")
    parser.add_argument("--max-candidates-per-s1", type=int, default=100, help="Max candidates per S1 entity budget (default: 100).")
    parser.add_argument("--max-bucket-size", type=int, default=100, help="Max bucket size for exact/prefix blocking (default: 100).")
    parser.add_argument("--max-token-doc-freq", type=int, default=2000, help="Max token document frequency for address/name tokens (default: 2000).")
    parser.add_argument("--tfidf-top-k", type=int, default=40, help="Top-K candidates per query in TF-IDF stage (default: 40).")
    parser.add_argument("--no-resume", action="store_true", help="Force recomputation of already existing chunks.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_training_shard(
        shard=args.shard,
        limit_s1=args.limit_s1,
        chunk_size=args.chunk_size,
        output_path=Path(args.output) if args.output else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_format=args.format,
        max_candidates_per_s1=args.max_candidates_per_s1,
        max_bucket_size=args.max_bucket_size,
        max_token_doc_freq=args.max_token_doc_freq,
        tfidf_top_k=args.tfidf_top_k,
        resume=not args.no_resume,
    )
