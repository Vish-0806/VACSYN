#!/usr/bin/env python
"""
SageMaker Training Entry Point for M1 + M2 Entity Resolution Pipeline.

This script orchestrates the existing M1 (blocking/candidate generation) and 
M2 (feature engineering + LightGBM training) components to run on SageMaker.

Architecture preserved exactly:
S1/S2/S3
→ M1 preprocessing
→ M1 multi-signal blocking
→ candidate pairs
→ M2 18 features
→ LightGBM
→ model artifact

No modifications to existing M1/M2 code.
"""

import sys
import time
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np

from src.business_entity_resolution.blocking.candidate_generator import (
    CandidateGenerator,
)
from src.business_entity_resolution.features.pair_features import build_pair_features
from src.business_entity_resolution.model.train import (
    prepare_ground_truth_mapping,
    assign_labels_to_candidates,
    entity_level_split,
    train_lightgbm_model,
    extract_model_features,
)

from s3_utils import (
    get_sagemaker_input_dir,
    get_sagemaker_model_dir,
    find_required_files,
    read_tsv,
    get_memory_usage_gb,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SageMaker training entry point for entity resolution"
    )
    parser.add_argument(
        "--max-candidates-per-s1",
        type=int,
        default=50,
        help="Maximum candidates per S1 entity (default: 50)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of LightGBM estimators (default: 500)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="LightGBM learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds (default: 50)",
    )
    return parser.parse_args()


def print_memory(label: str):
    """Print current memory usage."""
    mem = get_memory_usage_gb()
    if mem > 0:
        print(f"  [{label}] Memory: {mem:.2f} GB")
    sys.stdout.flush()


def main():
    print("=" * 70)
    print("SAGEMAKER TRAINING: M1 + M2 ENTITY RESOLUTION PIPELINE")
    print("=" * 70)
    print()

    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_memory("initial")

    # Parse arguments
    args = parse_args()
    print(f"Arguments: {vars(args)}")
    print()

    # Resolve SageMaker directories
    input_dir = get_sagemaker_input_dir()
    model_dir = get_sagemaker_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Model directory: {model_dir}")
    print()

    # Find required files
    print("Locating required training files...")
    try:
        files = find_required_files(input_dir)
        for key, path in files.items():
            print(f"  {key}: {path}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print()

    # Load ground truth once (small, fits in memory)
    print("1. Loading ground truth...")
    t0 = time.time()
    gt_mapping = prepare_ground_truth_mapping(files["gt"])
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  S1 entities with matches: {len(gt_mapping):,}")
    print_memory("after_gt")
    print()

    # Load S1 data once
    print("2. Loading S1 data...")
    t0 = time.time()
    s1_all = read_tsv(files["s1"])
    print(f"  Loaded in {time.time() - t0:.1f}s: {len(s1_all):,} rows")
    print_memory("after_s1")
    print()

    # Load S2 and S3 candidate pools once
    print("3. Loading S2 and S3 candidate pools...")
    t0 = time.time()
    s2_all = read_tsv(files["s2"])
    s3_all = read_tsv(files["s3"])
    print(f"  S2: {len(s2_all):,} rows")
    print(f"  S3: {len(s3_all):,} rows")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print_memory("after_s2_s3")
    print()

    # Get unique countries from S1
    countries = s1_all["country"].fillna("").astype(str).str.strip().str.upper().unique()
    countries = [c for c in countries if c]
    print(f"4. Found {len(countries)} country partitions: {sorted(countries)}")
    print()

    # Process each country sequentially
    all_candidates_list = []
    all_labeled_list = []
    all_features_list = []

    total_candidates = 0
    total_positives = 0
    total_negatives = 0

    for i, country in enumerate(sorted(countries), 1):
        print(f"[{i}/{len(countries)}] Processing country: {country}")

        # Filter to country
        s1_c = s1_all[s1_all["country"].fillna("").astype(str).str.strip().str.upper() == country]
        s2_c = s2_all[s2_all["country"].fillna("").astype(str).str.strip().str.upper() == country]
        s3_c = s3_all[s3_all["country"].fillna("").astype(str).str.strip().str.upper() == country]

        s1_count = len(s1_c)
        s2_count = len(s2_c)
        s3_count = len(s3_c)

        if s1_count == 0 or (s2_count == 0 and s3_count == 0):
            print(f"  Skipping - no data (S1={s1_count}, S2={s2_count}, S3={s3_count})")
            continue

        print(f"  S1: {s1_count:,} | S2: {s2_count:,} | S3: {s3_count:,}")
        print_memory(f"country_{country}_start")

        # M1: Candidate Generation (EXISTING function, unchanged)
        print("  Generating candidates...")
        t0 = time.time()
        generator = CandidateGenerator(max_candidates_per_s1=args.max_candidates_per_s1)
        candidates = generator.generate(s1_c, s2_c, s3_c)
        cand_time = time.time() - t0
        cand_count = len(candidates)
        total_candidates += cand_count
        print(f"  Generated {cand_count:,} candidates in {cand_time:.1f}s")
        print_memory(f"country_{country}_after_candidates")

        if cand_count == 0:
            print("  No candidates generated, skipping country")
            continue

        # M2: Feature Assembly (EXISTING function, unchanged)
        print("  Building features...")
        t0 = time.time()
        candidate_data = pd.concat([
            s2_c.assign(source="S2"),
            s3_c.assign(source="S3"),
        ], ignore_index=True)
        features = build_pair_features(candidates, s1_c, candidate_data)
        feat_time = time.time() - t0
        print(f"  Built {len(features):,} feature rows in {feat_time:.1f}s")
        print_memory(f"country_{country}_after_features")

        # Label Assignment (EXISTING function, unchanged)
        print("  Assigning labels...")
        t0 = time.time()
        labeled = assign_labels_to_candidates(candidates, gt_mapping)
        pos = int((labeled["label"] == 1).sum())
        neg = int((labeled["label"] == 0).sum())
        total_positives += pos
        total_negatives += neg
        print(f"  Labels: positive={pos:,}, negative={neg:,}, rate={pos/len(labeled)*100:.2f}%")
        print(f"  Labeled in {time.time() - t0:.1f}s")
        print_memory(f"country_{country}_after_labels")

        # Extract 18 model features + keep identity for split
        model_feats = extract_model_features(features)
        model_feats = model_feats.copy()
        model_feats["label"] = labeled["label"].values
        model_feats["s1_entity_id"] = candidates["s1_entity_id"].values

        # Accumulate
        all_candidates_list.append(candidates)
        all_labeled_list.append(labeled)
        all_features_list.append(model_feats)

        print(f"  Country {country} complete: {cand_count:,} candidates, {pos:,} pos, {neg:,} neg")
        print()

    # Combine all country shards
    print("5. Combining country results...")
    t0 = time.time()
    all_candidates = pd.concat(all_candidates_list, ignore_index=True) if all_candidates_list else pd.DataFrame()
    all_labeled = pd.concat(all_labeled_list, ignore_index=True) if all_labeled_list else pd.DataFrame()
    all_features = pd.concat(all_features_list, ignore_index=True) if all_features_list else pd.DataFrame()
    print(f"  Combined in {time.time() - t0:.1f}s")
    print(f"  Total candidates: {len(all_candidates):,}")
    print(f"  Total positive: {total_positives:,}")
    print(f"  Total negative: {total_negatives:,}")
    print(f"  Positive rate: {total_positives/len(all_labeled)*100:.2f}%" if len(all_labeled) > 0 else "  N/A")
    print(f"  Feature matrix shape: {all_features.shape}")
    print_memory("after_combine")
    print()

    if len(all_labeled) == 0:
        print("ERROR: No labeled candidates generated!")
        sys.exit(1)

    # Entity-level split (EXISTING function, unchanged)
    print("6. Entity-level train/validation split...")
    t0 = time.time()
    train_df, val_df = entity_level_split(
        all_labeled,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
    )
    print(f"  Split completed in {time.time() - t0:.1f}s")
    print(f"  Train: {len(train_df):,} candidates, {train_df['s1_entity_id'].nunique():,} S1 entities")
    print(f"  Val:   {len(val_df):,} candidates, {val_df['s1_entity_id'].nunique():,} S1 entities")

    # Verify no S1 overlap
    train_s1 = set(train_df["s1_entity_id"].unique())
    val_s1 = set(val_df["s1_entity_id"].unique())
    overlap = train_s1 & val_s1
    print(f"  S1 overlap: {len(overlap)} (must be 0)")
    assert len(overlap) == 0, f"S1 overlap detected: {overlap}"
    print("  ✓ Zero S1 overlap confirmed")
    print()

    # Prepare model features and labels for training
    print("7. Preparing model inputs...")
    model_features = all_features[["s1_entity_id"] + [c for c in all_features.columns if c not in ("s1_entity_id", "label")]].drop(columns=["s1_entity_id", "label"], errors="ignore")
    # Ensure column order matches MODEL_FEATURES
    from src.business_entity_resolution.model.train import MODEL_FEATURES
    model_features = model_features[MODEL_FEATURES]
    labels = all_labeled["label"]

    print(f"  Model features shape: {model_features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print()

    # Train LightGBM (EXISTING function, unchanged)
    print("8. Training LightGBM model...")
    print("  This may take several minutes...")
    sys.stdout.flush()
    t0 = time.time()

    lgbm_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": args.random_state,
        "n_jobs": -1,
        "verbose": 10,
        "force_col_wise": True,
        "early_stopping_rounds": args.early_stopping_rounds,
    }

    model = train_lightgbm_model(
        train_features_df=model_features.iloc[train_df.index],
        labels=labels.iloc[train_df.index],
        val_features_df=model_features.iloc[val_df.index],
        val_labels=labels.iloc[val_df.index],
        params=lgbm_params,
        model_save_path=None,  # We'll save manually to SM_MODEL_DIR
    )

    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s ({train_time/60:.1f} minutes)")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best validation logloss: {model.best_score['valid']['binary_logloss']:.6f}")
    print(f"  Training logloss: {model.best_score['training']['binary_logloss']:.6f}")
    print(f"  Features: {model.num_feature()}")
    print_memory("after_training")
    print()

    # Save model to SageMaker model directory
    print("9. Saving model artifact...")
    model_path = model_dir / "lightgbm_model.txt"
    model.save_model(str(model_path))
    model_size_mb = model_path.stat().st_size / (1024 ** 2)
    print(f"  Model saved to: {model_path}")
    print(f"  Model size: {model_size_mb:.2f} MB")
    print()

    # Verify model reload
    print("10. Verifying model reload...")
    import lightgbm as lgb
    loaded_model = lgb.Booster(model_file=str(model_path))
    print(f"  Loaded model features: {loaded_model.num_feature()}")
    assert loaded_model.num_feature() == 18, "Loaded model feature count mismatch!"
    print("  ✓ Model reload verified - 18 features confirmed")
    print()

    total_time = time.time() - start_time
    print("=" * 70)
    print("SAGEMAKER TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print("SUMMARY:")
    print(f"  Countries processed: {len(countries)}")
    print(f"  Total S1 entities: {len(s1_all):,}")
    print(f"  Total candidate pairs: {total_candidates:,}")
    print(f"  Positive candidates: {total_positives:,}")
    print(f"  Negative candidates: {total_negatives:,}")
    print(f"  Positive rate: {total_positives/total_candidates*100:.2f}%" if total_candidates > 0 else "  N/A")
    print(f"  Train candidates: {len(train_df):,}")
    print(f"  Validation candidates: {len(val_df):,}")
    print(f"  Train S1 entities: {train_df['s1_entity_id'].nunique():,}")
    print(f"  Validation S1 entities: {val_df['s1_entity_id'].nunique():,}")
    print(f"  S1 overlap: {len(overlap)} (confirmed zero)")
    print(f"  Feature matrix shape: {model_features.shape}")
    print(f"  Model features: 18 confirmed")
    print(f"  LightGBM best iteration: {model.best_iteration}")
    print(f"  Training logloss: {model.best_score['training']['binary_logloss']:.6f}")
    print(f"  Validation logloss: {model.best_score['valid']['binary_logloss']:.6f}")
    print(f"  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print_memory("final")
    print(f"  Model artifact: {model_path} ({model_size_mb:.2f} MB)")
    print("  Model reload: SUCCESS - 18 features confirmed")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()