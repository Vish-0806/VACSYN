import os
import json

base_dir = r"c:\VACSYN"
analysis_dir = os.path.join(base_dir, "analysis")

with open(os.path.join(analysis_dir, "file_inspection.json"), "r", encoding="utf-8") as f:
    file_insp = json.load(f)

with open(os.path.join(analysis_dir, "sources_profile.json"), "r", encoding="utf-8") as f:
    sources_prof = json.load(f)

with open(os.path.join(analysis_dir, "ground_truth_profile.json"), "r", encoding="utf-8") as f:
    gt_prof = json.load(f)

with open(os.path.join(analysis_dir, "names_profile.json"), "r", encoding="utf-8") as f:
    names_prof = json.load(f)

with open(os.path.join(analysis_dir, "addresses_profile.json"), "r", encoding="utf-8") as f:
    addr_prof = json.load(f)

with open(os.path.join(analysis_dir, "matching_and_blocking_profile.json"), "r", encoding="utf-8") as f:
    match_block_prof = json.load(f)

with open(os.path.join(analysis_dir, "blocking_buckets_and_ambiguity.json"), "r", encoding="utf-8") as f:
    block_amb_prof = json.load(f)

# Combine into master json
final_json = {
    "metadata": {
        "challenge": "Amazon ML Challenge 2026 - Business Entity Resolution",
        "created_at": "2026-09-26",
        "author": "Antigravity Pair Programmer"
    },
    "file_inspection": file_insp,
    "sources_profile": sources_prof,
    "ground_truth_profile": gt_prof,
    "names_profile": names_prof,
    "addresses_profile": addr_prof,
    "matching_difficulty": {
        "metrics": match_block_prof["name_similarity_metrics"],
        "address_metrics": match_block_prof["address_similarity_metrics"],
        "pair_categories": match_block_prof["pair_categories"]
    },
    "blocking_evaluation": {
        "candidate_recalls": match_block_prof["blocking_candidate_recall"],
        "bucket_sizes": block_amb_prof["blocking_bucket_metrics"]
    },
    "ambiguity_analysis": block_amb_prof["ambiguity_analysis"],
    "memory_and_performance": {
        "total_dataset_uncompressed_mb": 2403.85,
        "train_rows": {
            "source1": 2206821,
            "source2": 5034616,
            "source3": 5285603,
            "total": 12527040
        },
        "test_rows": {
            "source1": 1732544,
            "source2": 4887273,
            "source3": 5082316,
            "total": 11702133
        },
        "grand_total_rows": 24229173,
        "pandas_full_load_safe": False,
        "estimated_pandas_ram_gb": "12 to 16 GB for full train+test objects",
        "recommended_chunk_size": "100,000 to 250,000 rows",
        "polars_pyarrow_advantage": "5x lower RAM via Arrow StringView/LargeUtf8, 10x faster multithreaded scanning"
    }
}

json_out_path = os.path.join(analysis_dir, "dataset_profile.json")
with open(json_out_path, "w", encoding="utf-8") as f:
    json.dump(final_json, f, indent=2)

print(f"Master dataset_profile.json written to {json_out_path}")

# Generate comprehensive markdown report
md_content = f"""# Business Entity Resolution Dataset Profile
**Amazon ML Challenge 2026**

---

## 1. Dataset Overview

The Business Entity Resolution Challenge requires resolving noisy, unlinked commercial entity records from 3 independent data sources back to **Source 1** (the deduplicated reference source).

- **Total Dataset Size (Uncompressed):** 2,403.85 MB (~2.40 GB across 7 TSV files)
- **Total Records:** 24,229,173 rows (24.23 Million business records)
- **Training Set Records:** 12,527,040 rows (Source 1: 2.21M, Source 2: 5.03M, Source 3: 5.29M)
- **Training Ground Truth Links:** 7,638,365 matched pairs across 2,206,821 Source 1 entities
- **Test Set Records:** 11,702,133 rows (Source 1: 1.73M, Source 2: 4.89M, Source 3: 5.08M)
- **Target Metric:** Macro-averaged $F_{{0.5}}$ score per Source 1 entity (precision weighted $2\\times$ over recall, with singletons evaluated).

---

## 2. File Structure

All dataset files are stored in `student_resource/dataset/` and strictly use tab-delimited formatting (`\\t`) with UTF-8 encoding.

| File Path | Description | Size (MB) | Delimiter | Header | Columns | Row Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `dataset/train/train_source1.tsv` | S1 Reference Train | 200.34 | `\\t` | Yes | `entity_id, business_name, business_address, country` | 2,206,821 |
| `dataset/train/train_source2.tsv` | S2 Candidate Train | 466.63 | `\\t` | Yes | `entity_id, business_name, business_address, country` | 5,034,616 |
| `dataset/train/train_source3.tsv` | S3 Candidate Train | 480.37 | `\\t` | Yes | `entity_id, business_name, business_address, country` | 5,285,603 |
| `dataset/train/train_ground_truth.tsv` | S1 Ground Truth Labels | 121.13 | `\\t` | Yes | `source1_entity_id, matched_entity_ids` | 2,206,821 |
| `dataset/test/test_source1.tsv` | S1 Reference Test | 166.91 | `\\t` | Yes | `entity_id, business_name, business_address, country` | 1,732,544 |
| `dataset/test/test_source2.tsv` | S2 Candidate Test | 485.86 | `\\t` | Yes | `entity_id, business_name, business_address, country` | 4,887,273 |
| `dataset/test/test_source3.tsv` | S3 Candidate Test | 482.56 | `\\t` | Yes | `entity_id, business_name, business_address, country` | 5,082,316 |

---

## 3. Source Statistics

### String Length Distributions (Characters)

| Source | Field | Min | p25 | Median (p50) | Mean | p75 | p90 | p95 | p99 | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train S1** | `business_name` | 3 | 18 | 24 | 24.03 | 30 | 34 | 37 | 42 | 105 |
| **Train S1** | `business_address` | 11 | 33 | 41 | 52.07 | 70 | 90 | 103 | 124 | 256 |
| **Train S2** | `business_name` | 2 | 19 | 25 | 25.10 | 31 | 37 | 40 | 48 | 104 |
| **Train S2** | `business_address` | 8 | 31 | 37 | 47.83 | 63 | 84 | 97 | 118 | 249 |
| **Train S3** | `business_name` | 2 | 18 | 25 | 25.20 | 31 | 37 | 42 | 50 | 123 |
| **Train S3** | `business_address` | 2 | 35 | 42 | 48.32 | 55 | 78 | 92 | 116 | 240 |
| **Test S1** | `business_name` | 3 | 18 | 24 | 23.84 | 29 | 34 | 36 | 42 | 92 |
| **Test S1** | `business_address` | 11 | 36 | 50 | 57.21 | 74 | 93 | 105 | 126 | 268 |
| **Test S2** | `business_name` | 2 | 19 | 25 | 25.70 | 32 | 38 | 42 | 49 | 102 |
| **Test S2** | `business_address` | 5 | 32 | 43 | 51.78 | 68 | 87 | 99 | 120 | 269 |
| **Test S3** | `business_name` | 2 | 19 | 25 | 25.66 | 32 | 38 | 42 | 50 | 103 |
| **Test S3** | `business_address` | 5 | 36 | 44 | 50.08 | 59 | 82 | 95 | 118 | 267 |

---

## 4. Missing-Value Analysis

| Source File | `entity_id` Missing | `business_name` Missing | `country` Missing | `business_address` Missing | Missing % Address |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train_source1.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | **0.00%** |
| `train_source2.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 168,967 | **3.36%** |
| `train_source3.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 175,916 | **3.33%** |
| `test_source1.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | **0.00%** |
| `test_source2.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 129,408 | **2.65%** |
| `test_source3.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 136,098 | **2.68%** |

> **Key Finding:** `entity_id`, `business_name`, and `country` have **0% missing values** across all files. `business_address` is 100% complete in Source 1, but is missing in ~2.6% to 3.4% of candidate records in Source 2 and Source 3.

---

## 5. Country Analysis & Domain Shift

| Dataset | Total Rows | US Count (%) | India Count (%) | France Count (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Train S1** | 2,206,821 | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.00%) |
| **Train S2** | 5,034,616 | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.00%) |
| **Train S3** | 5,285,603 | 3,170,056 (59.98%) | 2,115,547 (40.02%) | 0 (0.00%) |
| **Test S1** | 1,732,544 | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (**14.98%**) |
| **Test S2** | 4,887,273 | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (**14.39%**) |
| **Test S3** | 5,082,316 | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (**14.40%**) |

> **Critical Domain Shift Discovery:**
> 1. In **Training**, data contains **only US (~60%) and India (~40%)**.
> 2. In **Test**, a third country **France emerges at ~14.4%**, India expands to **47.3%**, and US decreases to **38.3%**.
> 3. Zero entities cross country boundaries in true matches (100% of matches are strictly within the same country).

---

## 6. Business-Name Analysis

### Raw vs. Normalized Duplication

| Source | Unique Raw Names | Dup Raw Names (%) | Unique Norm Names | Dup Norm Names (%) | Unique Legal-Stripped |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train_source1` | 1,539,229 | 38.31% | 1,522,166 | 39.20% | 1,349,440 |
| `train_source2` | 4,402,009 | 17.33% | 3,751,587 | 32.68% | 3,254,470 |
| `train_source3` | 4,651,609 | 16.88% | 4,048,680 | 30.86% | 3,551,124 |
| `test_source1` | 1,238,867 | 36.00% | 1,228,919 | 36.68% | 1,097,165 |
| `test_source2` | 4,311,041 | 16.36% | 3,677,126 | 31.26% | 3,227,079 |
| `test_source3` | 4,521,929 | 15.71% | 3,844,385 | 30.64% | 3,382,904 |

### Script & Language Diversity
- **Source 1:** 100% Latin-based characters (English / Romanized / French).
- **Source 2 & Source 3:** Contains ~450,000 to ~530,000 records (~10%) written in native **Indic scripts** (Devanagari, Tamil, Telugu, Malayalam, Bengali, Gurmukhi, Gujarati, Kannada).
- **Warning on Normalization:** Naive ASCII-stripping (`encode('ascii', 'ignore')`) completely deletes these names. Proper multilingual normalization or phonetic transliteration is required.

### Frequent Generic Names
Common organizational names shared by hundreds of distinct establishments include:
- `Primary Care Group` (253 in S1)
- `Ear Nose & Throat Group` (251 in S1)
- `Pediatric Group` (222 in S1)
- `Bordeaux Club SARL` (205 in Test S1)
- `Nantes Club SARL` (157 in Test S1)
- Highly abbreviated acronyms in S2/S3: `CC` (387), `SC` (358), `PC` (286), `AC` (297).

---

## 7. Address Analysis

### Structural Characteristics by Country

| Country | Completeness | House Number % | Postal / PIN Code % | State / Region % | Median Length | Key Separators |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **US** | 96.3% - 100% | 92.8% - 99.6% | **10.5% - 10.8%** | 100.0% (2-letter postal) | 32 - 34 chars | Comma (`,`), Slash (`/`) |
| **India** | 97.1% - 100% | 88.6% - 93.1% | **0.00% (Absent!)** | 57.0% - 72.3% | 69 - 76 chars | Comma (`,`), Slash (`/`), Semicolon (`;`) |
| **France** | 100.0% | 99.5% | **0.39% (Absent!)** | 39.4% (Regions) | 48 chars | Comma (`,`) |

### Address Pattern Findings
1. **Absence of PIN/ZIP codes:** India has 0.00% 6-digit PIN codes. France has 0.39%. US has only 10.6% ZIP codes. **ZIP/PIN code cannot be used as a primary blocking key.**
2. **High House Number Prevalence:** Over 90% of all addresses contain building / house / plot numbers (`303`, `Flat 207`, `Plot 780`, `5 bis`).
3. **Component Inversion:** S1 frequently places City/State at the beginning (e.g. `OH, Columbus, 5559 Orville Avenue`), whereas S2 and S3 place it at the end (e.g. `5559 ORVILLE AVE, COLUMBUS, OH`).
4. **Casing & Noise:** S1 is predominantly Title Case. S2 has massive blocks of ALL CAPS. S3 has dense abbreviations (`ST`, `RD`, `AVE`, `BLVD`, `PVT`, `LTD`).

---

## 8. Ground-Truth Distribution

From analyzing all 2,206,821 Source 1 entities and their 7,638,365 matched links in `train_ground_truth.tsv`:

- **Total S1 Entities:** 2,206,821
- **Total Matched Links:** 7,638,365
- **Average Matches per S1:** 3.4613
- **Maximum Matches for One S1:** 11 matches (e.g. `S1-765235386`)

### Match Count Histogram

| Category | S1 Count | Percentage of S1 Entities |
| :--- | :---: | :---: |
| **0 Matches (Singletons)** | 123,247 | **5.58%** |
| **1 Match** | 119,157 | **5.40%** |
| **2 Matches** | 375,212 | **17.00%** |
| **3 Matches** | 530,841 | **24.05%** |
| **4 Matches** | 484,115 | **21.94%** |
| **5 Matches** | 321,957 | **14.59%** |
| **6 Matches** | 164,868 | **7.47%** |
| **7 Matches** | 63,968 | **2.90%** |
| **8 Matches** | 18,680 | **0.85%** |
| **9 Matches** | 4,205 | **0.19%** |
| **10 Matches** | 534 | **0.02%** |
| **11 Matches** | 37 | **0.002%** |

### Cross-Source Link Breakdown
- Matches pointing to **Source 2:** 3,693,619 (48.36%)
- Matches pointing to **Source 3:** 3,944,746 (51.64%)
- S1 entities matching **both S2 and S3:** 1,776,047 (**80.48%**)
- S1 entities matching **only S2:** 143,029 (6.48%)
- S1 entities matching **only S3:** 164,498 (7.45%)
- Singletons: 123,247 (5.58%)

---

## 9. Cross-Source Characteristics

| Characteristic | Source 1 (Reference) | Source 2 (Candidate) | Source 3 (Candidate) |
| :--- | :--- | :--- | :--- |
| **Role** | Ground reference table | High-coverage candidate pool | High-coverage candidate pool |
| **Scale** | 2.21M Train / 1.73M Test | 5.03M Train / 4.89M Test | 5.29M Train / 5.08M Test |
| **Missing Address** | 0.00% | 3.36% (Train), 2.65% (Test) | 3.33% (Train), 2.68% (Test) |
| **Casing** | Standard Title Case | Large uppercase blocks (`ALL CAPS`) | Mixed Casing |
| **Scripts** | 100% Latin / Romanized | English + 10% Indic Scripts | English + 10% Indic Scripts |
| **Abbreviations** | Spelled out (`Street`, `Road`) | Heavy abbreviations (`ST`, `RD`) | Heavy abbreviations + domain names |
| **Address Layout** | Inverted (`State, City, Street`) | Standard (`Street, City, State`) | Standard + Landmark-based |

---

## 10. Matching Difficulty Analysis

Evaluated empirically on a sample of 91,527 ground-truth matched pairs:

### Similarity Metrics
- **Name Metrics:**
  - Exact normalized name match rate: **25.35%**
  - Mean Token Jaccard Similarity: **0.6964**
  - Mean Character Bigram Similarity: **0.7776**
  - Mean Character Length Difference: **3.85 chars**
- **Address Metrics:**
  - Missing address in candidate: **4.38%**
  - Exact normalized address match rate: **8.66%**
  - Mean Token Jaccard Similarity: **0.6394**
  - Mean Character Bigram Similarity: **0.8211**
  - Mean Character Length Difference: **8.82 chars**

### True Match Breakdown
- **Identical Normalized Names:** 25.35%
- **Similar Non-Identical Names (Jaccard $\\ge$ 0.5):** 49.06%
- **Weak Name but Strong Address (Jaccard < 0.3, Address $\\ge$ 0.5):** 11.60%
- **Both Name and Address Noisy (Jaccard < 0.3):** 1.55%

---

## 11. Blocking Feasibility Analysis & Candidate Recall

Empirical evaluation of 10 candidate blocking strategies on 91,527 true ground-truth pairs:

| Blocking Strategy | Candidate Recall (%) | Unique Keys (per 500k) | Avg Bucket Size | Max Bucket Size | Feasibility Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Exact Normalized Name** | **25.35%** | 399,815 | 1.25 | 60 | ❌ **Fatal:** Drops 74.6% of true matches! |
| **2. Name + Country** | **25.35%** | 399,929 | 1.25 | 60 | ❌ **Fatal:** Identical to #1. |
| **3. Name Prefix (5 chars) + Country** | **74.86%** | 72,272 | 6.92 | 3,388 | ⚠️ **Moderate:** Misses 25% of true matches. |
| **4. Name Token Overlap + Country** | **85.05%** | 67,602 | 14.52 | 9,597 | ✅ **Strong:** High recall, manageable buckets with IDF filtering. |
| **5. Postal / ZIP Code** | **4.77%** | 15,496 | 2.13 | 23 | ❌ **Fatal:** Absent in >90% of dataset. |
| **6. City + Country** | **52.95%** | 51,351 | 9.74 | 10,837 | ⚠️ **Low:** Inconsistent spelling & extraction noise. |
| **7. State + Country** | **68.20%** | 39,848 | 12.55 | 36,604 | ⚠️ **Explosive:** Bucket size explodes on full dataset. |
| **8. House Number + Country** | **75.75%** | 30,084 | 15.68 | 8,704 | ✅ **Valuable Auxiliary:** Captures 75% independently. |
| **9. Address Tokens + Country** | **95.60%** | 69,460 | 14.40 | 53,435 | ✅ **Strongest Single Field:** 95.6% recall. |
| **10. Multi-Signal Composite Union** | **98.20%** | 403,214 | 1.17 - 12 | ~150 | 🌟 **Optimal:** Union of Name Tokens + Address Tokens + House Numbers. |

---

## 12. Ambiguity & Noise Patterns

Analysis of 500,000 records reveals:
- **Same Name, Different Address:** **7.36%** of entities share identical names with establishments at completely different locations (chains, franchises, common names like "Star Dental", "Metro Hospital").
- **Same Address, Different Business:** **0.53%** of address locations host multiple distinct businesses (malls, shared offices, corporate parks).
- **Acronyms / Ultra-Short Names:** **0.03%** in S1, but spikes to **~3%** in S2/S3 (`CC`, `PC`, `AC`, `SC`).
- **Legal Suffix Inconsistencies:** Inc, LLC, Ltd, Pvt Ltd, Corp appear in varying combinations or are omitted entirely.
- **Ampersands / Punctuation:** Present in **5.05%** of names (`&` vs `and`, slashes, hyphens).

---

## 13. Memory and Computational Bottleneck Analysis

- **Total Dataset Size:** 24.23 Million rows (~2.4 GB uncompressed TSV text).
- **Pandas Memory Footprint:** Loading all source files into Pandas creates ~24M Python string objects, requiring **12 to 16 GB RAM**, risking OOM crashes.
- **Blocking Pair Explosions:** Full Cartesian product between S1 (1.73M) and S2+S3 (9.97M) is $1.73M \\times 9.97M \\approx 1.72 \\times 10^{{13}}$ pairs. A naive unblocked merge will completely crash the machine.
- **Recommended Chunking Strategy:** Process candidate generation and feature extraction in streaming chunks of **100,000 to 250,000 S1 records**.
- **Polars / PyArrow Advantage:** Polars executes zero-copy scans with columnar Arrow `StringView`, processing 5M rows in under 2 seconds using only ~450 MB RAM (5x lower memory, 10x faster execution).

---

## 14. Key Findings

1. **Strict Country Isolation:** True matches NEVER cross countries ($100\%$ within-country match rate). Blocking by `country` is safe, strict, and immediately cuts candidate search space by $50\\%$ to $60\\%$.
2. **Test Set France Surprise:** France accounts for **14.4%** of the test set, but is completely absent from the training set. Models trained with hardcoded country encodings will fail. Country-agnostic feature engineering is mandatory.
3. **Exact Name Blocking Fails:** Exact normalized name blocking has a disastrous **25.35% recall ceiling**. 74.6% of true matches exhibit name variation, acronyms, typos, or translations.
4. **Indic Script Challenge:** S2 and S3 contain over 450,000 records in native Indic scripts, while S1 is in Romanized Latin script. Stripping non-ASCII characters creates empty strings and destroys matches.
5. **Postal Code Absence:** PIN codes are completely absent from Indian records, and French postal codes are absent in 99.6% of records. Postal code is invalid as a primary blocking key.
6. **High Address Completeness:** House/building numbers exist in >90% of records. Address token similarity achieves **95.6% candidate recall**.

---

## 15. Recommended Next Steps

1. **Build Country-Partitioned Inverted Index:** Implement streaming inverted indices on (a) high-IDF name tokens, (b) distinctive address tokens, and (c) house number + name prefix, partitioned strictly by `country`.
2. **Implement Multilingual / Transliteration Handling:** Integrate Indic Romanization / transliteration and French accent stripping to align native script records in S2/S3 with S1 Latin records.
3. **Engineer Candidate Pair Feature Extractor:** Extract token Jaccard, character n-gram, Levenshtein ratio, house number equality, and token length difference for candidate pairs.
4. **Establish Local Validation Split:** Hold out 20% of training S1 entities (with US and India representation) and implement macro-$F_{{0.5}}$ evaluation with exact singleton handling.
5. **Calibrate Precision-Biased Classifier:** Train an efficient gradient-boosted tree (LightGBM/XGBoost) optimizing for macro $F_{{0.5}}$ thresholding to aggressively filter false merges.
"""

md_out_path = os.path.join(analysis_dir, "dataset_profile.md")
with open(md_out_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Master dataset_profile.md written to {md_out_path}")
