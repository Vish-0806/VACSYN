#!/usr/bin/env python
"""
Full pipeline test for M2 Phase 2: End-to-end training with synthetic data.
"""

import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

from src.business_entity_resolution.model.train import (
    prepare_ground_truth_mapping,
    assign_labels_to_candidates,
    entity_level_split,
    train_lightgbm_model,
    extract_model_features,
)
from src.business_entity_resolution.features.pair_features import build_pair_features
from src.business_entity_resolution.blocking.candidate_generator import CandidateGenerator


def main():
    print("=== Creating synthetic data ===")
    np.random.seed(42)

    # Create S1, S2, S3 data
    s1_df = pd.DataFrame({
        'entity_id': [f'S1-{i:03d}' for i in range(50)],
        'business_name': [f'Company {i}' for i in range(50)],
        'business_address': [f'{i} Main St, City' for i in range(50)],
        'country': ['US'] * 50,
    })

    s2_df = pd.DataFrame({
        'entity_id': [f'S2-{i:03d}' for i in range(80)],
        'business_name': [f'Company {i}' for i in range(80)],
        'business_address': [f'{i} Main St, City' for i in range(80)],
        'country': ['US'] * 80,
    })

    s3_df = pd.DataFrame({
        'entity_id': [f'S3-{i:03d}' for i in range(70)],
        'business_name': [f'Company {i}' for i in range(70)],
        'business_address': [f'{i} Main St, City' for i in range(70)],
        'country': ['US'] * 70,
    })

    # Create ground truth: some S1 match S2, some match S3
    ground_truth = {}
    for i in range(20):
        ground_truth[f'S1-{i:03d}'] = [f'S2-{i:03d}']
    for i in range(20, 30):
        ground_truth[f'S1-{i:03d}'] = [f'S3-{i-20:03d}']
    for i in range(30, 35):
        ground_truth[f'S1-{i:03d}'] = [f'S2-{i:03d}', f'S3-{i-30:03d}']

    # Save ground truth to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id, matches in ground_truth.items():
            f.write(f'{s1_id}\t{",".join(matches)}\n')
        gt_path = f.name

    print(f'Ground truth saved to: {gt_path}')

    # Load ground truth
    gt_mapping = prepare_ground_truth_mapping(gt_path)
    print(f'Loaded ground truth for {len(gt_mapping)} S1 entities')

    # Generate candidates
    print('Generating candidates...')
    cg = CandidateGenerator(max_candidates_per_s1=20)
    candidates = cg.generate(s1_df, s2_df, s3_df)
    print(f'Generated {len(candidates)} candidate pairs')

    # Assign labels
    labeled = assign_labels_to_candidates(candidates, gt_mapping)
    print(f'Labeled {len(labeled)} candidates')
    print(f'Positive: {(labeled["label"]==1).sum()}, Negative: {(labeled["label"]==0).sum()}')

    # Entity-level split
    train_df, val_df = entity_level_split(labeled, val_ratio=0.2, random_state=42)
    print(f'Train: {len(train_df)} rows, {train_df["s1_entity_id"].nunique()} S1 entities')
    print(f'Val: {len(val_df)} rows, {val_df["s1_entity_id"].nunique()} S1 entities')

    # Verify no S1 overlap
    train_s1 = set(train_df['s1_entity_id'].unique())
    val_s1 = set(val_df['s1_entity_id'].unique())
    assert train_s1.isdisjoint(val_s1), 'S1 overlap detected!'
    print('Entity split verified: no S1 overlap')

    # Build features
    print('Building features...')
    candidate_data = pd.concat([s2_df.assign(source='S2'), s3_df.assign(source='S3')])
    features_df = build_pair_features(labeled, s1_df, candidate_data)
    print(f'Built features: {features_df.shape}')

    # Extract model features
    model_features = extract_model_features(features_df)
    print(f'Model features: {model_features.shape}, columns={list(model_features.columns)}')

    # Verify feature columns
    expected = ['name_token_jaccard', 'name_token_overlap_count', 'name_char_ngram_cosine',
                'name_edit_similarity', 'name_length_ratio', 'name_exact_match', 'name_first_token_match',
                'address_token_jaccard', 'address_token_overlap_count', 'address_house_number_match',
                'address_edit_similarity', 'address_length_ratio', 'address_missing',
                'country_match', 'source_is_s2', 'source_is_s3',
                'name_length_ratio_context', 'address_length_ratio_context']
    assert list(model_features.columns) == expected, 'Feature columns mismatch!'
    print('Feature columns verified!')

    # Train model
    print('Training LightGBM model...')
    model = train_lightgbm_model(
        train_features_df=model_features.iloc[train_df.index],
        labels=labeled.loc[train_df.index, 'label'],
        val_features_df=model_features.iloc[val_df.index],
        val_labels=labeled.loc[val_df.index, 'label'],
        params={'n_estimators': 20, 'verbose': -1},
    )

    print(f'Model trained: {model.num_feature()} features, best_iteration={model.best_iteration}')

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / 'test_model.txt'
        model.save_model(str(model_path))
        loaded = lgb.Booster(model_file=str(model_path))
        print(f'Model saved and loaded successfully! Features: {loaded.num_feature()}')

    # Cleanup
    Path(gt_path).unlink()

    print('\n=== FULL PIPELINE TEST PASSED ===')


if __name__ == '__main__':
    main()