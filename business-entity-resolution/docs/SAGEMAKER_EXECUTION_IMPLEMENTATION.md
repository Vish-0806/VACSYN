# SageMaker Execution Layer Implementation Report

## Overview

This document summarizes the implementation of the SageMaker execution layer for the M1 + M2 Entity Resolution pipeline. The implementation adds **only** the necessary orchestration code to run the existing frozen M1 (blocking/candidate generation) and M2 (feature engineering + LightGBM training) components on AWS SageMaker.

**Constraints strictly honored:**
- ✅ No modifications to M1 blocking/preprocessing/candidate generation
- ✅ No modifications to `pair_features.py`
- ✅ No modifications to `train.py`
- ✅ No changes to the 18 feature definitions
- ✅ No changes to `CandidateGenerator` behavior
- ✅ No changes to candidate schema
- ✅ No new model, embeddings, LLMs, APIs, RAG, databases, or Docker
- ✅ No streaming/out-of-core LightGBM implementation
- ✅ No TF-IDF setting alterations

---

## Files Created

### 1. `scripts/s3_utils.py` (128 lines)

Minimal S3/path utilities for the SageMaker execution layer.

| Function | Purpose |
|----------|---------|
| `resolve_s3_or_local_path(path)` | Normalize S3 URI or local path |
| `read_tsv(path, **kwargs)` | Read TSV from S3 or local via `s3fs` |
| `get_sagemaker_input_dir()` | Get `SM_CHANNEL_TRAIN` (default: `/opt/ml/input/data/train`) |
| `get_sagemaker_model_dir()` | Get `SM_MODEL_DIR` (default: `/opt/ml/model`) |
| `get_sagemaker_output_dir()` | Get `SM_OUTPUT_DATA_DIR` (default: `/opt/ml/output/data`) |
| `find_required_files(input_dir)` | Locate 4 required files; raise `FileNotFoundError` if missing |
| `get_memory_usage_gb()` | Current process memory in GB (uses `psutil`) |

**Dependencies added:** `s3fs`, `psutil` (in `requirements.txt`)

---

### 2. `scripts/sagemaker_train.py` (395 lines)

SageMaker training entry point. Orchestrates the **existing** M1/M2 functions without modification.

#### Key Characteristics

- **Single entry point** for SageMaker Training Job
- Reads configuration from SageMaker environment variables (`SM_CHANNEL_TRAIN`, `SM_MODEL_DIR`)
- Command-line arguments for hyperparameters (with sensible defaults matching existing code)
- **Country-by-country sequential processing** using existing `CandidateGenerator.generate()`
- Calls **existing** `build_pair_features()`, `assign_labels_to_candidates()`, `entity_level_split()`, `train_lightgbm_model()`, `extract_model_features()`
- Accumulates feature shards in memory, concatenates once, then trains
- Saves model to `SM_MODEL_DIR` for SageMaker artifact upload
- Comprehensive progress logging (country, counts, memory, timing, metrics)

#### Execution Flow

```
1. Parse args + resolve SageMaker directories
2. Locate required files in SM_CHANNEL_TRAIN:
   - train_source1.tsv
   - train_source2.tsv
   - train_source3.tsv
   - train_ground_truth.tsv
3. Load ground truth mapping (prepare_ground_truth_mapping) — EXISTING
4. Load S1, S2, S3 DataFrames (read_tsv) — NEW I/O only
5. Extract unique countries from S1
6. FOR each country (sorted, sequential):
   a. Filter S1/S2/S3 to country
   b. CandidateGenerator.generate(s1_c, s2_c, s3_c) — EXISTING M1
   c. build_pair_features(candidates, s1_c, s2_c+S3) — EXISTING M2
   d. assign_labels_to_candidates(candidates, gt_mapping) — EXISTING
   e. extract_model_features() + label + s1_entity_id — EXISTING
   f. Accumulate shards
7. pd.concat all shards → full feature matrix + labels
8. entity_level_split(all_labeled, val_ratio=0.2, random_state=42) — EXISTING
9. Verify zero S1 overlap
10. train_lightgbm_model(train_features, train_labels, val_features, val_labels) — EXISTING
11. Save model to SM_MODEL_DIR/lightgbm_model.txt
12. Verify reload (18 features confirmed)
```

#### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-candidates-per-s1` | 50 | Candidate budget per S1 entity |
| `--val-ratio` | 0.2 | Validation split ratio |
| `--random-state` | 42 | Random seed for reproducibility |
| `--n-estimators` | 500 | LightGBM estimators |
| `--learning-rate` | 0.05 | LightGBM learning rate |
| `--early-stopping-rounds` | 50 | Early stopping rounds |

---

## Files Modified

### `requirements.txt`

```diff
+ s3fs
+ psutil
```

---

## Test Results

| Test Suite | Result |
|------------|--------|
| All existing 201 tests (200 passed, 1 skipped) | ✅ **PASS** |
| `s3_utils.py` unit tests | ✅ **PASS** |
| `sagemaker_train.py` argument parsing tests | ✅ **PASS** |
| Import verification (all M1/M2 functions + new modules) | ✅ **PASS** |

**No existing tests modified.** All 200 existing tests continue to pass.

---

## Memory/Runtime Analysis

### Memory Profile (Estimated)

| Stage | Memory Estimate |
|-------|-----------------|
| S1 DataFrame (2.2M rows × ~4 cols) | ~200 MB |
| S2 DataFrame (~1M rows) | ~100 MB |
| S3 DataFrame (~1M rows) | ~100 MB |
| Ground truth mapping | ~50 MB |
| Per-country candidate pool + indices | ~500 MB - 2 GB (varies by country) |
| Full feature matrix at training | ~2-8 GB (depends on candidate count) |
| LightGBM Dataset objects | ~2-4 GB |

**Recommended SageMaker instance:** `ml.r5.4xlarge` (128 GB) or `ml.r5.8xlarge` (256 GB)

### Runtime Concerns

| Concern | Severity | Mitigation |
|---------|----------|------------|
| TF-IDF blocker on large country (US) | High | `max_features=40_000`; ensure large instance; cannot modify |
| `assign_labels_to_candidates` uses `apply()` row-wise | Medium | Slow on 10M+ rows; cannot modify; accept longer runtime |
| Full feature matrix concatenation | Medium | Single allocation; same as current `full_training.py` |
| No checkpointing | Medium | Failure restarts from scratch; acceptable for single training run |
| S3 read latency | Low | `s3fs` streams via HTTP range requests |

---

## SageMaker Infrastructure Requirements

| Component | Specification |
|-----------|---------------|
| **Training Job Type** | `ScriptProcessor` or `Estimator` with `entry_point="scripts/sagemaker_train.py"` |
| **Instance Type** | `ml.r5.4xlarge` (128 GB) or `ml.r5.8xlarge` (256 GB) |
| **Base Image** | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-cpu-py310-ubuntu20.04-sagemaker` (or equivalent Python 3.10 CPU image) |
| **Input Channel** | `SM_CHANNEL_TRAIN=s3://amazon-ml-2026-vishal-entity-resolution/train/` |
| **Output** | Model artifact auto-uploaded from `SM_MODEL_DIR` (`/opt/ml/model`) |
| **IAM Role** | `SageMakerFullAccess` + `S3ReadAccess` to the bucket |

**No custom Docker image required.** Dependencies install via `pip install -r requirements.txt`.

---

## Git Diff Summary

```
 business-entity-resolution/requirements.txt | 2 ++
 business-entity-resolution/scripts/s3_utils.py        (new, 128 lines)
 business-entity-resolution/scripts/sagemaker_train.py (new, 395 lines)
```

**Zero modifications to any `src/` files.** All M1/M2 logic called unchanged.

---

## Next Steps (Not Implemented)

1. **Create SageMaker Training Job** — Use `sagemaker_train.py` as entry point
2. **Upload training data to S3** — Already at `s3://amazon-ml-2026-vishal-entity-resolution/train/`
3. **Launch training job** — Via SageMaker console, CLI, or SDK
4. **Monitor CloudWatch logs** — Progress output includes country-by-country metrics
5. **Retrieve model artifact** — From `SM_MODEL_DIR` after job completion

---

*Implementation complete. Awaiting SageMaker job launch.*