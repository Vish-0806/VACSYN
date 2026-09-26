"""
Main Candidate Generator and Blocking Coordinator.

This is the primary blocking interface feeding directly into pairwise feature extraction
and ML scoring. The candidate set produced here corresponds exactly to candidate_pairs.tsv.

Interface Contract:
    Output Candidate Pairs Schema:
        - s1_entity_id: Identifier of reference Source 1 entity (e.g. 'S1-00001').
        - candidate_entity_id: Identifier of candidate entity (e.g. 'S2-00047', 'S3-00812').
        - candidate_source: Origin of candidate ('S2' or 'S3').

Ownership: Member 1 (Preprocessing & Blocking).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from business_entity_resolution.preprocessing.normalize import (
    normalize_business_name,
    tokenize_business_name,
)
from business_entity_resolution.preprocessing.address import normalize_address
from business_entity_resolution.blocking.exact import (
    ExactIndex,
    AddressBlocker,
    generate_exact_name_key,
    generate_name_prefix_key,
)
from business_entity_resolution.blocking.fuzzy import NameTokenBlocker
from business_entity_resolution.blocking.tfidf import SparseTFIDFBlocker

logger = logging.getLogger(__name__)


def _standardize_columns(df: pd.DataFrame, default_source: str) -> pd.DataFrame:
    """
    Standardize DataFrame column names and fill missing values.
    """
    df = df.copy()
    col_map = {}
    for col in df.columns:
        c_lower = col.strip().lower()
        if c_lower in ("entity_id", "s1_entity_id", "s2_entity_id", "s3_entity_id", "id"):
            col_map[col] = "entity_id"
        elif c_lower in ("business_name", "name", "company_name"):
            col_map[col] = "business_name"
        elif c_lower in ("business_address", "address", "addr"):
            col_map[col] = "business_address"
        elif c_lower in ("country", "country_code", "cntry"):
            col_map[col] = "country"
        elif c_lower in ("source", "candidate_source"):
            col_map[col] = "source"

    df.rename(columns=col_map, inplace=True)

    for required in ["entity_id", "business_name", "business_address", "country"]:
        if required not in df.columns:
            df[required] = ""

    if "source" not in df.columns:
        df["source"] = default_source

    df["entity_id"] = df["entity_id"].astype(str).str.strip()
    df["business_name"] = df["business_name"].fillna("").astype(str).str.strip()
    df["business_address"] = df["business_address"].fillna("").astype(str).str.strip()
    df["country"] = df["country"].fillna("").astype(str).str.strip().str.upper()
    df["source"] = df["source"].fillna(default_source).astype(str).str.strip()

    return df


class CandidateGenerator:
    """
    Coordinates country-partitioned multi-signal candidate generation.
    Unites deterministic exact/prefix blocking, address/house number blocking,
    rarest-first name token blocking, and character n-gram sparse TF-IDF retrieval.
    """

    def __init__(
        self,
        max_candidates_per_s1: int = 50,
        enable_tfidf: bool = True,
        tfidf_top_k: int = 20,
        tfidf_min_similarity: float = 0.35,
    ) -> None:
        """
        Initialize candidate generator.

        Args:
            max_candidates_per_s1: Maximum candidate budget per Source 1 entity.
            enable_tfidf: Whether to execute sparse TF-IDF character n-gram retrieval.
            tfidf_top_k: Top-K candidates retrieved per query in TF-IDF stage.
            tfidf_min_similarity: Minimum cosine similarity threshold for TF-IDF.
        """
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.enable_tfidf = enable_tfidf
        self.tfidf_top_k = tfidf_top_k
        self.tfidf_min_similarity = tfidf_min_similarity

    def generate(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate candidate pairs using multi-signal blocking strategies.

        Args:
            s1_df: Source 1 reference records [entity_id, business_name, business_address, country].
            s2_df: Source 2 candidate pool records.
            s3_df: Source 3 candidate pool records.

        Returns:
            DataFrame containing candidate pairs with columns:
            ['s1_entity_id', 'candidate_entity_id', 'candidate_source']
        """
        columns = ["s1_entity_id", "candidate_entity_id", "candidate_source"]
        if s1_df is None or s1_df.empty:
            return pd.DataFrame(columns=columns)

        s1_clean = _standardize_columns(s1_df, default_source="S1")
        s2_clean = _standardize_columns(s2_df, default_source="S2") if s2_df is not None and not s2_df.empty else pd.DataFrame()
        s3_clean = _standardize_columns(s3_df, default_source="S3") if s3_df is not None and not s3_df.empty else pd.DataFrame()

        # Combine candidate pools
        pool_parts = [p for p in [s2_clean, s3_clean] if not p.empty]
        if not pool_parts:
            return pd.DataFrame(columns=columns)

        candidate_pool = pd.concat(pool_parts, ignore_index=True)
        if candidate_pool.empty:
            return pd.DataFrame(columns=columns)

        # Build candidate source map
        cand_source_map = dict(zip(candidate_pool["entity_id"], candidate_pool["source"]))

        results: List[Tuple[str, str, str]] = []

        # Process country partitions strictly isolated
        countries = s1_clean["country"].unique()

        for country in countries:
            if not country:
                continue

            s1_country = s1_clean[s1_clean["country"] == country]
            pool_country = candidate_pool[candidate_pool["country"] == country]

            if s1_country.empty or pool_country.empty:
                continue

            part_pairs = self._generate_partition(
                country=country,
                s1_part=s1_country,
                pool_part=pool_country,
                cand_source_map=cand_source_map,
            )
            results.extend(part_pairs)

        if not results:
            return pd.DataFrame(columns=columns)

        out_df = pd.DataFrame(results, columns=columns)
        out_df.drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"], inplace=True)
        return out_df

    def _generate_partition(
        self,
        country: str,
        s1_part: pd.DataFrame,
        pool_part: pd.DataFrame,
        cand_source_map: Dict[str, str],
    ) -> List[Tuple[str, str, str]]:
        """
        Generate candidates for a single country partition using multi-signal blocking.
        """
        cand_ids = pool_part["entity_id"].tolist()
        cand_names = pool_part["business_name"].tolist()
        cand_addrs = pool_part["business_address"].tolist()

        s1_ids = s1_part["entity_id"].tolist()
        s1_names = s1_part["business_name"].tolist()
        s1_addrs = s1_part["business_address"].tolist()

        # 1. Exact and Prefix Name Index
        exact_index = ExactIndex(max_bucket_size=30)
        for cid, cname in zip(cand_ids, cand_names):
            k_exact = generate_exact_name_key(cname, country)
            if k_exact:
                exact_index.add(k_exact, cid)
            k_pref = generate_name_prefix_key(cname, country, prefix_len=5)
            if k_pref:
                exact_index.add(k_pref, cid)

        # 2. Address & House Number Blocker
        addr_blocker = AddressBlocker(max_token_doc_freq=500, max_candidates_per_entity=30)
        for cid, cname, caddr in zip(cand_ids, cand_names, cand_addrs):
            addr_blocker.add_entity(
                entity_id=cid,
                address=caddr,
                normalized_name=cname,
                country=country,
            )

        # 3. Name Token Blocker (Rarest-first, Generic Exclusions)
        token_blocker = NameTokenBlocker(max_token_doc_freq=500, max_candidates_per_entity=30)
        for cid, cname in zip(cand_ids, cand_names):
            c_toks = tokenize_business_name(cname)
            token_blocker.add_entity(entity_id=cid, tokens=c_toks, country=country)

        # 4. Sparse TF-IDF Blocker (Character N-gram Cosine Dot Product)
        tfidf_matches: Dict[int, List[str]] = {}
        if self.enable_tfidf and len(cand_names) > 0:
            tfidf_blocker = SparseTFIDFBlocker(
                ngram_range=(3, 4),
                max_features=40_000,
                min_df=2,
            )
            tfidf_blocker.fit_transform_corpus(cand_names, cand_ids)
            query_tfidf_res = tfidf_blocker.query_candidate_ids(
                s1_names,
                top_k=self.tfidf_top_k,
                min_similarity=self.tfidf_min_similarity,
            )
            for idx, c_list in enumerate(query_tfidf_res):
                tfidf_matches[idx] = c_list

        partition_pairs: List[Tuple[str, str, str]] = []

        # Query candidates for each S1 entity and union with priority
        for idx in range(len(s1_ids)):
            s1_id = s1_ids[idx]
            s1_name = s1_names[idx]
            s1_addr = s1_addrs[idx]

            # Ordered dictionary to maintain candidate priority and deduplicate
            cand_candidates: Dict[str, None] = {}

            # Priority 1: Exact Name Key
            k_exact = generate_exact_name_key(s1_name, country)
            if k_exact:
                for cid in exact_index.query(k_exact):
                    cand_candidates[cid] = None
                    if len(cand_candidates) >= self.max_candidates_per_s1:
                        break

            # Priority 2: Address Multi-Path Blocker (House number + Prefix, Distinctive Addr Tokens)
            if len(cand_candidates) < self.max_candidates_per_s1:
                addr_cands = addr_blocker.query_candidates(
                    address=s1_addr,
                    normalized_name=s1_name,
                    country=country,
                )
                for cid in addr_cands:
                    cand_candidates[cid] = None
                    if len(cand_candidates) >= self.max_candidates_per_s1:
                        break

            # Priority 3: Name Prefix Key
            if len(cand_candidates) < self.max_candidates_per_s1:
                k_pref = generate_name_prefix_key(s1_name, country, prefix_len=5)
                if k_pref:
                    for cid in exact_index.query(k_pref):
                        cand_candidates[cid] = None
                        if len(cand_candidates) >= self.max_candidates_per_s1:
                            break

            # Priority 4: Name Token Blocker (Rarest-first Token Inverted Index)
            if len(cand_candidates) < self.max_candidates_per_s1:
                s1_toks = tokenize_business_name(s1_name)
                token_cands = token_blocker.query_candidates(
                    query_tokens=s1_toks,
                    country=country,
                )
                for cid in token_cands:
                    cand_candidates[cid] = None
                    if len(cand_candidates) >= self.max_candidates_per_s1:
                        break

            # Priority 5: Sparse TF-IDF Top-K Retrieval
            if len(cand_candidates) < self.max_candidates_per_s1 and idx in tfidf_matches:
                for cid in tfidf_matches[idx]:
                    cand_candidates[cid] = None
                    if len(cand_candidates) >= self.max_candidates_per_s1:
                        break

            # Append valid candidate pairs
            for cid in cand_candidates.keys():
                src = cand_source_map.get(cid, "S2")
                partition_pairs.append((s1_id, cid, src))

        return partition_pairs


def _load_data(data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    """Helper to load DataFrame from path or return DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, sep="\t", dtype=str)


def generate_candidates(
    s1_data: Any,
    s2_data: Any,
    s3_data: Any,
    output_path: Optional[Path] = None,
    max_candidates_per_s1: int = 50,
) -> pd.DataFrame:
    """
    Generate final candidate pairs for ML scoring conforming to Contract 1.

    Args:
        s1_data: Source 1 DataFrame or file path.
        s2_data: Source 2 DataFrame or file path.
        s3_data: Source 3 DataFrame or file path.
        output_path: Optional file path to stream/write candidate_pairs.tsv.
        max_candidates_per_s1: Candidate budget per Source 1 entity.

    Returns:
        DataFrame containing columns:
        - s1_entity_id
        - candidate_entity_id
        - candidate_source
    """
    s1_df = _load_data(s1_data)
    s2_df = _load_data(s2_data) if s2_data is not None else pd.DataFrame()
    s3_df = _load_data(s3_data) if s3_data is not None else pd.DataFrame()

    generator = CandidateGenerator(max_candidates_per_s1=max_candidates_per_s1)
    cand_df = generator.generate(s1_df, s2_df, s3_df)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cand_df.to_csv(out_p, sep="\t", index=False)
        logger.info("Saved %d candidate pairs to %s", len(cand_df), out_p)

    return cand_df
