# Methodology

## 1. Problem Understanding
<!-- TODO: Describe problem formulation, reference Source 1, noisy Source 2 and Source 3 candidate sources, macro F0.5 precision-biased evaluation metric, and singleton handling. -->

## 2. Data Preprocessing
<!-- TODO: Describe Unicode-aware normalization, multilingual Indic script handling, case standardization, legal suffix extraction, address cleaning, house number extraction, and missing address imputation. -->

## 3. Candidate Generation / Blocking
<!-- TODO: Describe country partitioning, inverted index construction on distinctive name and address tokens, sparse TF-IDF retrieval, candidate recall ceiling analysis, and candidate set size bounds. -->

## 4. Feature Engineering
<!-- TODO: Detail pairwise string similarity features (token Jaccard, character n-grams, edit distances), house number exact match flags, street similarity, source indicators, and length difference metrics. -->

## 5. Model Architecture
<!-- TODO: Describe LightGBM gradient-boosted decision tree architecture, hyperparameter selection rationale, and binary classification framing for pair matching. -->

## 6. Training Strategy
<!-- TODO: Explain train/validation partitioning, hard negative sampling from candidate generation, handling class imbalance, and loss formulation. -->

## 7. Threshold Selection
<!-- TODO: Document grid search methodology for probability thresholding optimizing macro F0.5, singleton protection rules, and multi-match precision-recall tradeoff. -->

## 8. Validation
<!-- TODO: Present validation results (candidate recall, validation precision, recall, and macro F0.5) and generalization across countries (US, India, unseen France). -->

## 9. Error Analysis
<!-- TODO: Analyze false merges (precision errors), missed matches, singleton failure cases, and language/script failure modes. -->

## 10. Final Submission Generation
<!-- TODO: Document generation and validation of matching_results.tsv and candidate_pairs.tsv ensuring strict format adherence and absence of illegal IDs. -->

## 11. Computational Considerations
<!-- TODO: Outline chunked streaming processing, memory consumption bounds, runtime benchmarks, and zero-copy columnar scanning optimizations. -->

## 12. Limitations
<!-- TODO: Discuss edge cases, potential failure modes on unseen countries or scripts, and future directions for improvement. -->
