#!/usr/bin/env python
"""
Phase 2.7: Full real-data LightGBM training on complete training dataset.
"""

import sys
import time
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import psutil
import os

# Add src to path
sys.path.insert(0, 'src')

from src.business_entity_resolution.model.train import (
    prepare_ground_truth_mapping,
    assign_labels_to_candidates,
    entity_level_split,
    train_lightgbm_model,
    extract_model_features,
)
from src.business_entity_resolution.features.pair_features import build_pair_features
from src.business_entity_resolution.blocking.candidate_generator import generate_candidates


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def main():
    print("=" * 60)
    print("PHASE 2.7: FULL REAL-DATA LIGHTGBM TRAINING")
    print("=" * 60)
    print()

    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Initial memory: {get_memory_usage():.2f} GB")
    print()

    # Data paths
    data_dir = Path(r'C:\Users\Vishal S Naik\MyProjects\VACSYN\student_resource\dataset\train\extracted\student_resource\dataset\train')
    s1_path = data_dir / 'train_source1.tsv'
    s2_path = data_dir / 'train_source2.tsv'
    s3_path = data_dir / 'train_source3.tsv'
    gt_path = data_dir / 'train_ground_truth.tsv'

    # Verify files exist
    for p in [s1_path, s2_path, s3_path, gt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print("Files verified:")
    print(f"  S1: {s1_path.stat().st_size / (1024**3):.2f} GB")
    print(f"  S2: {s2_path.stat().st_size / (1024**3):.2f} GB")
    print(f"  S3: {s3_path.stat().st_size / (1024**3):.2f} GB")
    print(f"  GT: {gt_path.stat().st_size / (1024**3):.2f} GB")
    print()

    # 1. Load ground truth
    print("1. Loading ground truth...")
    t0 = time.time()
    gt_mapping = prepare_ground_truth_mapping(str(gt_path))
    print(f"  Ground truth loaded in {time.time() - t0:.1f}s")
    print(f"  S1 entities with matches: {len(gt_mapping):,}")
    print(f"  Memory: {get_memory_usage():.2f} GB")
    print()

    # 2. Load all S1 data
    print("2. Loading S1 data...")
    t0 = time.time()
    s1_df = pd.read_csv(s1_path, sep='\t', dtype=str)
    print(f"  S1 loaded in {time.time() - t0:.1f}s: {len(s1_df):,} rows")
    print(f"  Memory: {get_memory_usage():.2f} GB")
    print()

    # 4. Generate candidates using the full pipeline (chunked)
    print("4. Generating candidates (full dataset)...")
    print("  This will take a while - generating candidates for ~2.2M S1 entities...")
    sys.stdout.flush()
    t0 = time.time()
    candidates = generate_candidates(
        s1_data=s1_path,
        s2_data=s2_path,
        s3_data=s3_path,
        max_candidates_per_s1=50,
    )
    print(f"  Candidates generated in {time.time() - t0:.1f}s")
    print(f"  Total candidate pairs: {len(candidates):,}")
    print(f"  Memory: {get_memory_usage():.2f} GB")
    print()
    sys.stdout.flush()

    # 5. Load ground truth for label assignment
    print("5. Assigning labels...")
    t0 = time.time()
    labeled = assign_labels_to_candidates(candidates, gt_mapping)
    pos = (labeled['label'] == 1).sum()
    neg = (labeled['label'] == 0).sum()
    print(f"  Labels assigned in {time.time() - t0:.1f}s")
    print(f"  Total candidates: {len(labeled):,}")
    print(f"  Positive: {pos:,} ({pos/len(labeled)*100:.2f}%)")
    print(f"  Negative: {neg:,} ({neg/len(labeled)*100:.2f}%)")
    print(f"  Memory: {get_memory_usage():.2f} GB")
    print()

    # 6. Entity-level split
    print("6. Entity-level train/validation split...")
    t0 = time.time()
    train_df, val_df = entity_level_split(labeled, val_ratio=0.2, random_state=42)
    print(f"  Split completed in {time.time() - t0:.1f}s")
    print(f"  Train: {len(train_df):,} candidates, {train_df['s1_entity_id'].nunique():,} S1")
    print(f"  Val: {len(val_df):,} candidates, {val_df['s1_entity_id'].nunique():,} S1")
    print()

    # Verify no S1 overlap
    train_s1 = set(train_df['s1_entity_id'].unique())
    val_s1 = set(val_df['s1_entity_id'].unique())
    overlap = train_s1 & val_s1
    print(f"  Train S1: {len(train_s1):,}")
    print(f"  Val S1: {len(val_s1):,}")
    print(f"  S1 overlap: {len(overlap)} (should be 0)")
    assert len(overlap) == 0, f"S1 overlap detected: {overlap}"
    print("  ✓ Zero S1 overlap confirmed")
    print()

    # 7. Load S2 and S3 data for feature building
    print("7. Loading S2 and S3 data for feature building...")
    t0 = time.time()
    s2_df = pd.read_csv(s2_path, sep='\t', dtype=str)
    s3_df = pd.read_csv(s3_path, sep='\t', dtype=str)
    print(f"  S2 loaded: {len(s2_df):,} rows")
    print(f"  S3 loaded: {len(s3_df):,} rows")
    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"  Memory: {get_memory_usage():.2f} GB")
    print()

    # 8. Build features
    print("8. Building features...")
    t0 = time.time()
    candidate_data = pd.concat([s2_df.assign(source='S2'), s3_df.assign(source='S3')])
    features_df = build_pair_features(labeled, s1_df, candidate_data)
    print(f"  Features built in {time.time() - t0:.1f}s")
    print(f"  Feature df shape: {features_df.shape}")
    print(f"  Memory: {get_memory_usage():.2f} GB")
    print()

    # 9. Extract model features
    print("9. Extracting model features...")
    t0 = time.time()
    model_features = extract_model_features(features_df)
    print(f"  Extracted in {time.time() - t0:.1f}s")
    print(f"  Model features shape: {model_features.shape}")
    print()

    # 10. Verify 18 features
    expected_features = [
        'name_token_jaccard', 'name_token_overlap_count', 'name_char_ngram_cosine',
        'name_edit_similarity', 'name_length_ratio', 'name_exact_match', 'name_first_token_match',
        'address_token_jaccard', 'address_token_overlap_count', 'address_house_number_match',
        'address_edit_similarity', 'address_length_ratio', 'address_missing',
        'country_match', 'source_is_s2', 'source_is_s3',
        'name_length_ratio_context', 'address_length_ratio_context'
    ]
    assert list(model_features.columns) == expected_features, 'Feature columns mismatch!'
    print("✓ Exactly 18 model features confirmed")
    print(f"  Features: {model_features.columns.tolist()}")
    print()

    # 11. Train LightGBM
    print("11. Training LightGBM model...")
    print("  This may take several minutes...")
    t0 = time.time()

    model = train_lightgbm_model(
        train_features_df=model_features.iloc[train_df.index],
        labels=labeled.loc[train_df.index, 'label'],
        val_features_df=model_features.iloc[val_df.index],
        val_labels=labeled.loc[val_df.index, 'label'],
        params={
            'n_estimators': 100,  # Reduced for faster completion
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 10,  # Log every 10 iterations
            'force_col_wise': True,
        },
    )

    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s ({train_time/60:.1f} minutes)")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best validation logloss: {model.best_score['valid']['binary_logloss']:.6f}")
    print(f"  Features: {model.num_feature()}")
    print(f"  Memory: {get_memory_usage():.2f} GB")
    print()

    # 12. Save model
    print("12. Saving model artifact...")
    model_path = Path('models/lightgbm_model.txt')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    model_size = model_path.stat().st_size / (1024 ** 2)
    print(f"  Model saved to: {model_path}")
    print(f"  Model size: {model_size:.2f} MB")
    print()

    # 13. Verify model reload
    print("13. Verifying model reload...")
    loaded_model = lgb.Booster(model_file=str(model_path))
    print(f"  Loaded model features: {loaded_model.num_feature()}")
    assert loaded_model.num_feature() == 18, "Loaded model feature count mismatch!"
    print("  ✓ Model reload verified - 18 features confirmed")
    print()

    total_time = time.time() - start_time
    print("=" * 60)
    print("FULL REAL-DATA TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("SUMMARY:")
    print(f"  Total S1 entities in dataset: {len(s1_path.read_text().splitlines()) - 1:,}")
    print(f"  Total candidate pairs: {len(candidates):,}")
    print(f"  Positive candidates: {pos:,}")
    print(f"  Negative candidates: {neg:,}")
    print(f"  Positive rate: {pos/len(labeled)*100:.2f}%")
    print(f"  Train S1 entities: {train_df['s1_entity_id'].nunique():,}")
    print(f"  Validation S1 entities: {val_df['s1_entity_id'].nunique():,}")
    print(f"  Train candidate rows: {len(train_df):,}")
    print(f"  Validation candidate rows: {len(val_df):,}")
    print(f"  S1 overlap: {len(overlap)} (confirmed zero)")
    print("  Feature matrix shape: {}".format(features_df.shape))
    print("  Model features: 18 confirmed")
    print("  LightGBM best iteration: {}".format(model.best_iteration))
    print("  Training logloss: {:.6f}".format(model.best_score['training']['binary_logloss']))
    print("  Validation logloss: {:.6f}".format(model.best_score['valid']['binary_logloss']))
    print("  Total runtime: {:.1f}s ({:.1f} min)".format(time.time() - start_time, (time.time() - start_time)/60))
    print("  Peak memory: {:.2f} GB".format(get_memory_usage()))
    print("  Model artifact: models/lightgbm_model.txt ({:.2f} MB)".format(model_size))
    print("  Model reload: SUCCESS - 18 features confirmed")
    print()
    print("REAL FULL MODEL TRAINED: YES")
    print("=" * 60)


if __name__ == '__main__':
    main()