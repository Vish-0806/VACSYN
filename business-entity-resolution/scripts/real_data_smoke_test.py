#!/usr/bin/env python
"""
Phase 2.6: Small real-data smoke test using actual training data.
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

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
from src.business_entity_resolution.blocking.candidate_generator import CandidateGenerator


def main():
    print("=== Phase 2.6: Real-Data Smoke Test ===")
    print()

    # Data paths
    data_dir = Path(r'C:\Users\Vishal S Naik\MyProjects\VACSYN\student_resource\dataset\train\extracted\student_resource\dataset\train')
    s1_path = data_dir / 'train_source1.tsv'
    s2_path = data_dir / 'train_source2.tsv'
    s3_path = data_dir / 'train_source3.tsv'
    gt_path = data_dir / 'train_ground_truth.tsv'

    print('Loading data...')
    # Load ground truth first
    gt_mapping = prepare_ground_truth_mapping(str(gt_path))
    print('Ground truth loaded: {} S1 entities with matches'.format(len(gt_mapping)))

    # Get S1 IDs that have ground truth
    s1_with_gt = list(gt_mapping.keys())
    print('S1 with GT: {}'.format(len(s1_with_gt)))

    # Sample 500 S1 entities from those with ground truth
    np.random.seed(42)
    sampled_s1_ids = np.random.choice(s1_with_gt, size=min(500, len(s1_with_gt)), replace=False)
    print('Sampled S1 entities: {}'.format(len(sampled_s1_ids)))

    # Load S1 data for sampled entities
    s1_rows = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if parts[0] in sampled_s1_ids:
                s1_rows.append(parts)
                if len(s1_rows) >= len(sampled_s1_ids):
                    break
    s1_df = pd.DataFrame(s1_rows, columns=header)
    print('Loaded S1: {} rows'.format(len(s1_df)))

    # Load S2 and S3 data needed for the sampled S1 entities
    # We need to load S2/S3 entities that are in ground truth for these S1 + some distractors
    needed_s2 = set()
    needed_s3 = set()
    for s1_id in sampled_s1_ids:
        for m in gt_mapping.get(s1_id, []):
            if m.startswith('S2-'):
                needed_s2.add(m)
            elif m.startswith('S3-'):
                needed_s3.add(m)

    print('Needed S2: {}, Needed S3: {}'.format(len(needed_s2), len(needed_s3)))

    # Load S2
    s2_rows = []
    distractors_s2 = 0
    with open(s2_path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if parts[0] in needed_s2:
                s2_rows.append(parts)
            elif distractors_s2 < 500:  # Add some distractors
                s2_rows.append(parts)
                distractors_s2 += 1
            if len(s2_rows) >= len(needed_s2) + 500:
                break
    s2_df = pd.DataFrame(s2_rows, columns=header)
    print('Loaded S2: {} rows'.format(len(s2_df)))

    # Load S3
    s3_rows = []
    distractors_s3 = 0
    with open(s3_path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if parts[0] in needed_s3:
                s3_rows.append(parts)
            elif distractors_s3 < 500:
                s3_rows.append(parts)
                distractors_s3 += 1
            if len(s3_rows) >= len(needed_s3) + 500:
                break
    s3_df = pd.DataFrame(s3_rows, columns=header)
    print('Loaded S3: {} rows'.format(len(s3_df)))

    # Generate candidates
    print('Generating candidates...')
    cg = CandidateGenerator(max_candidates_per_s1=40)
    candidates = cg.generate(s1_df, s2_df, s3_df)
    print('Generated {} candidate pairs'.format(len(candidates)))

    # Assign labels
    labeled = assign_labels_to_candidates(candidates, gt_mapping)
    pos = (labeled['label'] == 1).sum()
    neg = (labeled['label'] == 0).sum()
    print('Labeled {} candidates: Positive={}, Negative={}, Rate={:.2f}%'.format(
        len(labeled), pos, neg, pos/len(labeled)*100))

    # Entity-level split
    train_df, val_df = entity_level_split(labeled, val_ratio=0.2, random_state=42)
    print('Train: {} candidates, {} S1'.format(len(train_df), train_df['s1_entity_id'].nunique()))
    print('Val: {} candidates, {} S1'.format(len(val_df), val_df['s1_entity_id'].nunique()))

    # Build features
    print('Building features...')
    candidate_data = pd.concat([s2_df.assign(source='S2'), s3_df.assign(source='S3')])
    features_df = build_pair_features(labeled, s1_df, candidate_data)
    print('Feature df shape: {}'.format(features_df.shape))

    # Extract model features
    model_features = extract_model_features(features_df)
    print('Model features shape: {}'.format(model_features.shape))
    print('Model features columns: {}'.format(list(model_features.columns)))

    # Verify 18 features
    expected = [
        'name_token_jaccard', 'name_token_overlap_count', 'name_char_ngram_cosine',
        'name_edit_similarity', 'name_length_ratio', 'name_exact_match', 'name_first_token_match',
        'address_token_jaccard', 'address_token_overlap_count', 'address_house_number_match',
        'address_edit_similarity', 'address_length_ratio', 'address_missing',
        'country_match', 'source_is_s2', 'source_is_s3',
        'name_length_ratio_context', 'address_length_ratio_context'
    ]
    assert list(model_features.columns) == expected, 'Feature columns mismatch!'
    print('Exactly 18 model features confirmed!')

    # Train LightGBM
    print('Training LightGBM...')
    model = train_lightgbm_model(
        train_features_df=model_features.iloc[train_df.index],
        labels=labeled.loc[train_df.index, 'label'],
        val_features_df=model_features.iloc[val_df.index],
        val_labels=labeled.loc[val_df.index, 'label'],
        params={'n_estimators': 100, 'verbose': -1},
    )
    print('LightGBM training SUCCESS! Features: {}, Best iter: {}'.format(model.num_feature(), model.best_iteration))

    print()
    print('=== SMOKE TEST PASSED ===')


if __name__ == '__main__':
    main()