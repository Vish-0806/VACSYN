"""
Sparse TF-IDF and character n-gram candidate retrieval.

Responsible for scalable sparse matrix retrieval:
- Character n-gram and word-level TF-IDF vectorization with sublinear scaling.
- Top-K cosine similarity retrieval using sparse matrix multiplications.
- Country-partitioned candidate retrieval avoiding dense matrix allocations.

Important:
    Do NOT build giant dense matrices in memory.
    Uses scipy.sparse CSR matrix representations and top-k sparse dot products.

Ownership: Member 1 (Preprocessing & Blocking).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class SparseTFIDFBlocker:
    """
    Scalable sparse TF-IDF blocker for text matching.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        max_features: int = 50_000,
        min_df: int = 2,
    ) -> None:
        """
        Initialize sparse TF-IDF blocker.

        Args:
            ngram_range: Boundary of character n-grams (within word boundaries).
            max_features: Upper limit on vocabulary dimension.
            min_df: Minimum document frequency for n-gram inclusion.
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.corpus_matrix: Optional[sparse.csr_matrix] = None
        self.candidate_ids: List[str] = []

    def fit_transform_corpus(
        self,
        texts: List[str],
        candidate_ids: Optional[List[str]] = None,
    ) -> sparse.csr_matrix:
        """
        Fit vectorizer on candidate texts and return sparse representation.

        Args:
            texts: List of candidate business texts (e.g. name or name + address).
            candidate_ids: Optional list of entity IDs corresponding to texts.

        Returns:
            Fitted sparse CSR matrix of shape (N_candidates, V).
        """
        if not texts:
            self.corpus_matrix = sparse.csr_matrix((0, 0), dtype=np.float32)
            self.candidate_ids = []
            return self.corpus_matrix

        effective_min_df = self.min_df if len(texts) >= 5 else 1
        self.vectorizer.set_params(min_df=effective_min_df)

        try:
            # Fit vectorizer and compute L2-normalized sparse TF-IDF matrix
            self.corpus_matrix = self.vectorizer.fit_transform(texts)
        except ValueError as e:
            if "no terms remain" in str(e).lower() and effective_min_df > 1:
                self.vectorizer.set_params(min_df=1)
                try:
                    self.corpus_matrix = self.vectorizer.fit_transform(texts)
                except ValueError:
                    self.corpus_matrix = sparse.csr_matrix((len(texts), 0), dtype=np.float32)
            else:
                self.corpus_matrix = sparse.csr_matrix((len(texts), 0), dtype=np.float32)

        if candidate_ids is not None:
            self.candidate_ids = list(candidate_ids)
        else:
            self.candidate_ids = [str(i) for i in range(len(texts))]

        return self.corpus_matrix

    def query_top_k(
        self,
        query_texts: List[str],
        top_k: int = 20,
        min_similarity: float = 0.3,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Retrieve top-k candidate indices for each query text using sparse matrix multiplication.

        Since both matrices are L2-normalized, cosine similarity is the sparse dot product:
            Similarity = Query @ Corpus.T

        Args:
            query_texts: Reference S1 texts to query.
            top_k: Maximum candidate matches to return per query.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            Dictionary mapping query index to list of (candidate_index, similarity_score).
        """
        if (
            self.corpus_matrix is None
            or self.corpus_matrix.shape[0] == 0
            or self.corpus_matrix.shape[1] == 0
            or not query_texts
        ):
            return {i: [] for i in range(len(query_texts))}

        # Transform queries to sparse representation
        query_matrix = self.vectorizer.transform(query_texts)

        # Sparse dot product: shape (N_queries, N_candidates)
        similarity_matrix = query_matrix.dot(self.corpus_matrix.T)

        results: Dict[int, List[Tuple[int, float]]] = {}

        # Efficiently extract top-k entries per row directly from CSR row pointers
        for i in range(similarity_matrix.shape[0]):
            row_start = similarity_matrix.indptr[i]
            row_end = similarity_matrix.indptr[i + 1]

            if row_start == row_end:
                results[i] = []
                continue

            indices = similarity_matrix.indices[row_start:row_end]
            data = similarity_matrix.data[row_start:row_end]

            # Filter above minimum similarity
            mask = data >= min_similarity
            if not np.any(mask):
                results[i] = []
                continue

            valid_indices = indices[mask]
            valid_scores = data[mask]

            if len(valid_scores) > top_k:
                # Use argpartition to find top_k largest elements in O(N) time
                top_pos = np.argpartition(valid_scores, -top_k)[-top_k:]
                # Sort only the top_k elements
                top_pos = top_pos[np.argsort(-valid_scores[top_pos])]
                top_cand_indices = valid_indices[top_pos]
                top_cand_scores = valid_scores[top_pos]
            else:
                sort_order = np.argsort(-valid_scores)
                top_cand_indices = valid_indices[sort_order]
                top_cand_scores = valid_scores[sort_order]

            results[i] = [
                (int(idx), float(score))
                for idx, score in zip(top_cand_indices, top_cand_scores)
            ]

        return results

    def query_candidate_ids(
        self,
        query_texts: List[str],
        top_k: int = 20,
        min_similarity: float = 0.3,
    ) -> List[List[str]]:
        """
        Retrieve top-k candidate entity IDs for each query text.

        Returns:
            List of candidate ID lists corresponding to query_texts.
        """
        top_k_indices = self.query_top_k(query_texts, top_k=top_k, min_similarity=min_similarity)
        all_candidate_ids: List[List[str]] = []

        for i in range(len(query_texts)):
            cand_list = [
                self.candidate_ids[cand_idx]
                for cand_idx, _ in top_k_indices.get(i, [])
                if cand_idx < len(self.candidate_ids)
            ]
            all_candidate_ids.append(cand_list)

        return all_candidate_ids


class CountrySparseTFIDFBlocker:
    """
    Manages country-partitioned sparse TF-IDF blockers.
    Guarantees strict country isolation and cuts RAM requirements by partitioning.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        max_features_per_country: int = 40_000,
        top_k: int = 20,
        min_similarity: float = 0.35,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features_per_country = max_features_per_country
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.country_blockers: Dict[str, SparseTFIDFBlocker] = {}

    def fit_country_corpus(
        self,
        country: str,
        texts: List[str],
        candidate_ids: List[str],
    ) -> None:
        """Fit sparse TF-IDF index for a specific country."""
        c = country.strip().upper()
        blocker = SparseTFIDFBlocker(
            ngram_range=self.ngram_range,
            max_features=self.max_features_per_country,
        )
        blocker.fit_transform_corpus(texts, candidate_ids)
        self.country_blockers[c] = blocker

    def query(
        self,
        country: str,
        query_texts: List[str],
    ) -> List[List[str]]:
        """Query top candidates within the given country partition."""
        c = country.strip().upper()
        blocker = self.country_blockers.get(c)
        if blocker is None:
            return [[] for _ in query_texts]
        return blocker.query_candidate_ids(
            query_texts,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
        )
