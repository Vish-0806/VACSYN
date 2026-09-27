# SageMaker Execution Analysis: M1 + M2 Pipeline Migration

## Executive Summary

This document analyzes the minimal execution-layer changes required to run the existing real-data training pipeline on AWS SageMaker using training files already uploaded to:

```
s3://amazon-ml-2026-vishal-entity-resolution/train/
```

**Constraints honored:**
- M1 blocking/candidate-generation code is FROZEN
- No modifications to: `src/business_entity_resolution/blocking/`, `candidate_generator.py`, M1 preprocessing, blocking logic, candidate schemas, thresholds, TF-IDF config
- No modifications to 18 feature definitions or `pair_features.py`
- No modifications to `train.py`
- No new streaming/out-of-core ML algorithm
- No embeddings, LLMs, APIs, RAG, databases, Docker, or new ML models

---

## 1. Existing Functions Reusable Unchanged

| Module | Function/Class | Reusable As-Is |
|--------|----------------|----------------|
| `blocking.candidate_generator` | `CandidateGenerator.generate(s1_df, s2_df, s3_df)` | ✅ Yes — already partitions by country internally |
| `blocking.candidate_generator` | `generate_candidates(s1_data, s2_data, s3_data, ...)` | ✅ Yes — accepts DataFrames or paths |
| `features.pair_features` | `build_pair_features(candidate_pairs, s1_data, candidate_data)` | ✅ Yes — pure function, returns 18-feature DataFrame |
| `features.pair_features` | `PairFeatureExtractor.extract_features(...)` | ✅ Yes — internal, called by `build_pair_features` |
| `model.train` | `prepare_ground_truth_mapping(gt_path)` | ✅ Yes — reads TSV, returns dict |
| `model.train` | `assign_labels_to_candidates(candidates, gt_mapping)` | ✅ Yes — pure DataFrame operation |
| `model.train` | `entity_level_split(candidates, val_ratio, random_state)` | ✅ Yes — pure DataFrame operation |
| `model.train` | `extract_model_features(features_df)` | ✅ Yes — selects 18 columns |
| `model.train` | `train_lightgbm_model(...)` | ✅ Yes — requires full train/val DataFrames in memory |
| `preprocessing.normalize` | `normalize_business_name`, `tokenize_business_name` | ✅ Yes — pure functions |
| `preprocessing.address` | `normalize_address` | ✅ Yes — pure functions |
| `blocking.exact/fuzzy/tfidf` | All classes/functions | ✅ Yes — internal to CandidateGenerator |

---

## 2. New Execution/Orchestration Files Required

| File | Purpose |
|------|---------|
| `scripts/sagemaker_train.py` | **Single entry point** for SageMaker Training Job. Reads `SM_CHANNEL_TRAIN`, `SM_MODEL_DIR`. Orchestrates country-by-country loop, calls existing functions, accumulates feature shards, invokes `train_lightgbm_model()`. |
| `scripts/s3_utils.py` | Minimal helpers: `read_tsv_s3(uri) -> pd.DataFrame`, `write_parquet_s3(df, uri)`, `list_countries_s3(prefix)` — no boto3 in core logic. |

**No other new files.** No custom Docker, no streaming ML, no modified M1/M2.

---

## 3. Country-by-Country Processing Without Modifying M1

The existing `CandidateGenerator.generate()` **already iterates countries internally** (lines 145-163 in `candidate_generator.py`):

```python
countries = s1_clean["country"].unique()
for country in countries:
    s1_country = s1_clean[s1_clean["country"] == country]
    pool_country = candidate_pool[candidate_pool["country"] == country]
    part_pairs = self._generate_partition(country, s1_country, pool_country, ...)
```

**Execution-layer approach in `sagemaker_train.py`:**

```python
# 1. Load full S1 once (~2.2M rows × ~4 cols ≈ 200 MB)
s1_all = pd.read_csv("s3://.../train_source1.tsv", sep="\t", dtype=str)
countries = s1_all["country"].unique()

# 2. Load S2/S3 once (candidate pools)
s2_all = pd.read_csv("s3://.../train_source2.tsv", sep="\t", dtype=str)
s3_all = pd.read_csv("s3://.../train_source3.tsv", sep="\t", dtype=str)

# 3. Load ground truth once
gt_mapping = prepare_ground_truth_mapping("s3://.../train_ground_truth.tsv")

# 4. For each country:
for country in countries:
    s1_c = s1_all[s1_all["country"] == country]
    s2_c = s2_all[s2_all["country"] == country]
    s3_c = s3_all[s3_all["country"] == country]
    
    # Call EXISTING M1 - no changes
    candidates = CandidateGenerator(max_candidates_per_s1=50).generate(s1_c, s2_c, s3_c)
    
    # Call EXISTING M2 feature assembly - no changes
    cand_data = pd.concat([s2_c.assign(source="S2"), s3_c.assign(source="S3")])
    features = build_pair_features(candidates, s1_c, cand_data)
    
    # Assign labels using EXISTING function
    labeled = assign_labels_to_candidates(candidates, gt_mapping)
    
    # Extract 18 features + label + s1_entity_id for split
    model_feats = extract_model_features(features)
    model_feats["label"] = labeled["label"].values
    model_feats["s1_entity_id"] = candidates["s1_entity_id"].values
    
    # Accumulate shard
    shards.append(model_feats)
```

**Key point:** M1 is not modified. We filter DataFrames *before* calling `generate()`. The existing country-partitioning logic inside `CandidateGenerator` will see only one country and process it.

---

## 4. Handling Feature Data for LightGBM Without New Algorithm

**Constraint:** `train_lightgbm_model()` (frozen) requires:
- `train_features_df`: full training feature matrix (pd.DataFrame, 18 cols)
- `labels`: full training labels (pd.Series)
- `val_features_df`, `val_labels`: same for validation

**Solution within constraints:**

1. **Accumulate all country feature shards** into a list of DataFrames during the country loop
2. **After loop:** `all_features = pd.concat(shards, ignore_index=True)`
3. **Entity-level split** using EXISTING `entity_level_split()` on the combined labeled candidates
4. **Index into combined feature matrix** using train/val indices (as `full_training.py` does)
5. **Call EXISTING `train_lightgbm_model()`** with the full matrices

```python
# After country loop:
all_labeled = pd.concat(labeled_shards, ignore_index=True)
all_features = pd.concat(feature_shards, ignore_index=True)

# EXISTING split
train_df, val_df = entity_level_split(all_labeled, val_ratio=0.2, random_state=42)

# EXISTING feature extraction + indexing (same as full_training.py)
model_features = extract_model_features(all_features)
model = train_lightgbm_model(
    train_features_df=model_features.iloc[train_df.index],
    labels=all_labeled.loc[train_df.index, "label"],
    val_features_df=model_features.iloc[val_df.index],
    val_labels=all_labeled.loc[val_df.index, "label"],
    params={...},
    model_save_path=Path(SM_MODEL_DIR) / "lightgbm_model.txt",
)
```

**Memory implication:** Full feature matrix + labels must fit in RAM at training time. This is identical to current `full_training.py` — the difference is we build it incrementally per-country instead of all-at-once.

---

## 5. Minimum SageMaker Infrastructure Required

| Component | Specification |
|-----------|---------------|
| **SageMaker Training Job** | `ScriptProcessor` or `Estimator` with `entry_point="scripts/sagemaker_train.py"` |
| **Instance Type** | `ml.r5.4xlarge` (128 GB RAM) or `ml.r5.8xlarge` (256 GB RAM) — accommodates full feature matrix in memory |
| **Base Image** | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-cpu-py310-ubuntu20.04-sagemaker` (or equivalent Python 3.10 CPU image) — **no custom Docker needed**; `requirements.txt` installs via `pip install -r requirements.txt` |
| **Input Channel** | `SM_CHANNEL_TRAIN=s3://amazon-ml-2026-vishal-entity-resolution/train/` |
| **Output** | Model artifact auto-uploaded from `SM_MODEL_DIR` (`/opt/ml/model`) |
| **IAM Role** | `SageMakerFullAccess` + `S3ReadAccess` to the bucket |
| **Requirements** | Current `requirements.txt` + `s3fs` (for `pd.read_csv("s3://...")`) |

**No:** Pipe mode, custom containers, batch transform, hosting endpoints, processing jobs.

---

## 6. Remaining Memory/Runtime Risks

| Risk | Severity | Mitigation Within Constraints |
|------|----------|-------------------------------|
| **Full feature matrix OOM at training** | High | Use `ml.r5.8xlarge` (256 GB); current `full_training.py` runs locally — if it works locally, it works on larger instance |
| **TF-IDF blocker memory per large country (US)** | High | `SparseTFIDFBlocker(max_features=40_000)` builds 40K × N sparse matrix. Cannot modify. Mitigation: ensure instance has enough RAM; TF-IDF runs per-country inside `generate()` |
| **S2/S3 full pools in memory during country loop** | Medium | S2/S3 loaded once (~1-2M rows each). Filter per-country creates views. Acceptable on 128+ GB |
| **`pd.concat` of all shards at end** | Medium | Single allocation of final matrix. Same as current `full_training.py` |
| **S3 read latency for 3 large TSVs** | Low | `pd.read_csv("s3://...")` uses `s3fs` → streams via HTTP range requests; acceptable |
| **`assign_labels_to_candidates` uses `apply()` row-wise** | Medium | Slow on 10M+ rows. Cannot modify. Accept longer runtime |
| **No checkpointing** | Medium | If job fails mid-country-loop, restarts from scratch. Acceptable for single training run |

---

## Proposed Execution Flow

```
SageMaker Training Job (ml.r5.4xlarge or ml.r5.8xlarge)
├─ Input: s3://amazon-ml-2026-vishal-entity-resolution/train/
│   ├─ train_source1.tsv
│   ├─ train_source2.tsv
│   ├─ train_source3.tsv
│   └─ train_ground_truth.tsv
│
├─ Step 1: Load S1, S2, S3, GT from S3 (full DataFrames)
│
├─ Step 2: For each country in S1["country"].unique():
│   ├─ Filter S1, S2, S3 to country
│   ├─ CandidateGenerator.generate(s1_c, s2_c, s3_c) → candidates
│   ├─ build_pair_features(candidates, s1_c, s2_c + s3_c) → features
│   ├─ assign_labels_to_candidates(candidates, gt_mapping) → labeled
│   ├─ extract_model_features(features) + label + s1_entity_id → shard
│   └─ Append shard to lists
│
├─ Step 3: pd.concat all shards → full feature matrix + labeled pairs
│
├─ Step 4: entity_level_split(labeled) → train/val indices
│
├─ Step 5: train_lightgbm_model(train_features, train_labels, val_features, val_labels)
│
└─ Step 6: Save model to SM_MODEL_DIR (/opt/ml/model/lightgbm_model.txt)
```

---

## Files to Create (Upon Approval)

1. `scripts/s3_utils.py` — ~30 lines, S3 read helpers
2. `scripts/sagemaker_train.py` — ~150 lines, orchestration entry point
3. Update `requirements.txt` — add `s3fs`

**No modifications to any existing `src/` files.**

---

*Generated for review. Awaiting approval before implementation.*