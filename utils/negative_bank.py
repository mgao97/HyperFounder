"""
Hard Negative Bank Module for Challenge 2.

Stores weak but valid sub-hypergraphs for hard negative sampling.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional

import torch


class HardNegativeBank:
    """
    Bank for storing weak but valid sub-hypergraphs for hard negative sampling.
    
    Subgraphs are categorized by:
    - Quality tier (low, medium, high)
    - Domain
    - Size characteristics
    
    This enables:
    - Diverse hard negative sampling
    - Domain-aware negatives
    - Quality-controlled sampling
    """

    def __init__(
        self,
        max_size: int = 1000,
        num_tiers: int = 3,
        num_domains: int = 8,
        sampling_strategy: str = "quality_weighted",
    ):
        """
        Args:
            max_size: Maximum number of subgraphs to store
            num_tiers: Number of quality tiers
            num_domains: Number of domains
            sampling_strategy: 'random', 'quality_weighted', 'fifo'
        """
        self.max_size = max_size
        self.num_tiers = num_tiers
        self.num_domains = num_domains
        self.sampling_strategy = sampling_strategy

        # Organized storage: tier -> domain -> list of entries
        self.bank: Dict[int, Dict[int, deque]] = {
            tier: {domain: deque(maxlen=max(max_size // (num_tiers * num_domains), 10))
                    for domain in range(num_domains)}
            for tier in range(num_tiers)
        }
        
        # Quality tiers boundaries
        self.tier_boundaries = [0.0, 0.25, 0.5, 1.0]

    def _get_tier(self, quality_score: float) -> int:
        """Map quality score to tier (0 = lowest, num_tiers-1 = highest)."""
        for tier in range(self.num_tiers - 1, -1, -1):
            if quality_score >= self.tier_boundaries[tier]:
                return tier
        return 0

    def add(
        self,
        subhypergraph_data: dict,
        quality_score: float,
        domain_id: int,
        hedg_meta: dict | None = None,
    ) -> bool:
        """
        Add a sub-hypergraph to the bank.

        Args:
            subhypergraph_data: Dictionary containing subhypergraph info
            quality_score: Quality score [0, 1] (higher = structurally richer)
            domain_id: Domain identifier
            hedg_meta: Optional HEDG-level metadata for negatives from the
                HEDG-weighted sampler, e.g. ``{"avg_similarity": float,
                "fallback_rate": float, "num_negatives": int}``.

        Returns:
            True if added, False if rejected
        """
        tier = self._get_tier(quality_score)

        if domain_id >= self.num_domains:
            domain_id = 0

        hedg_meta = hedg_meta or {}
        hedg_avg_sim = float(hedg_meta.get("avg_similarity", 0.0))
        hedg_fallback = float(hedg_meta.get("fallback_rate", 0.0))
        # Combined "hardness" proxy: 1 - quality (low quality = hard)
        # boosted if HEDG similarity is high (high overlap = hard negatives).
        hardness_proxy = (1.0 - quality_score) + 0.5 * min(hedg_avg_sim / max(hedg_avg_sim + hedg_fallback + 1e-6, 1.0), 1.0)
        entry = {
            "data": subhypergraph_data,
            "quality_score": quality_score,
            "domain_id": domain_id,
            "tier": tier,
            "hedg_avg_similarity": hedg_avg_sim,
            "hedg_fallback_rate": hedg_fallback,
            "hardness_proxy": float(max(0.0, min(1.0, hardness_proxy))),
            "hedg_meta": hedg_meta,
        }

        self.bank[tier][domain_id].append(entry)
        self._trim()

        return True

    def _trim(self):
        """Remove oldest entries if bank exceeds max size."""
        total_size = sum(
            len(self.bank[t][d])
            for t in range(self.num_tiers)
            for d in range(self.num_domains)
        )
        
        while total_size > self.max_size:
            # Remove from lowest tier first
            for tier in range(self.num_tiers):
                for domain in range(self.num_domains):
                    if len(self.bank[tier][domain]) > 0:
                        self.bank[tier][domain].popleft()
                        total_size -= 1
                        if total_size <= self.max_size:
                            return

    def sample(
        self,
        batch_size: int,
        domain_id: Optional[int] = None,
        min_quality: Optional[float] = None,
        max_quality: Optional[float] = None,
    ) -> List[dict]:
        """
        Sample hard negatives from the bank.
        
        Args:
            batch_size: Number of samples to return
            domain_id: Optional domain filter
            min_quality: Minimum quality score
            max_quality: Maximum quality score
        
        Returns:
            List of sampled entries
        """
        if self.sampling_strategy == "random":
            return self._sample_random(batch_size, domain_id, min_quality, max_quality)
        elif self.sampling_strategy == "quality_weighted":
            return self._sample_quality_weighted(batch_size, domain_id, min_quality, max_quality)
        elif self.sampling_strategy == "fifo":
            return self._sample_fifo(batch_size, domain_id, min_quality, max_quality)
        else:
            return self._sample_random(batch_size, domain_id, min_quality, max_quality)

    def _collect_candidates(
        self,
        domain_id: Optional[int] = None,
        min_quality: Optional[float] = None,
        max_quality: Optional[float] = None,
    ) -> List[tuple]:
        """Collect all candidate entries."""
        candidates = []
        
        for tier in range(self.num_tiers):
            for domain in range(self.num_domains):
                if domain_id is not None and domain != domain_id:
                    continue
                    
                for entry in self.bank[tier][domain]:
                    quality = entry["quality_score"]
                    if min_quality is not None and quality < min_quality:
                        continue
                    if max_quality is not None and quality > max_quality:
                        continue
                    candidates.append((entry, tier, domain))
        
        return candidates

    def _sample_random(
        self,
        batch_size: int,
        domain_id: Optional[int],
        min_quality: Optional[float],
        max_quality: Optional[float],
    ) -> List[dict]:
        """Random sampling."""
        candidates = self._collect_candidates(domain_id, min_quality, max_quality)
        
        if len(candidates) <= batch_size:
            return [c[0] for c in candidates]
        
        selected = random.sample(candidates, batch_size)
        return [c[0] for c in selected]

    def _sample_quality_weighted(
        self,
        batch_size: int,
        domain_id: Optional[int],
        min_quality: Optional[float],
        max_quality: Optional[float],
    ) -> List[dict]:
        """Sample with quality-weighted probability.

        Combines (a) inverse-quality (low quality = harder) and
        (b) HEDG hardness proxy (high HEDG similarity = harder) into
        a single sampling weight so that the bank naturally favours
        subgraphs that produced the most informative negatives.
        """
        candidates = self._collect_candidates(domain_id, min_quality, max_quality)

        if len(candidates) <= batch_size:
            return [c[0] for c in candidates]

        weights = []
        for entry, tier, domain in candidates:
            hardness = float(entry.get("hardness_proxy", 0.0))
            q = float(entry.get("quality_score", 0.5))
            # Base inverse quality (as before)
            base = 1.0 / (q + 0.1)
            tier_boost = (self.num_tiers - tier) / self.num_tiers
            hedg_boost = 1.0 + hardness
            weights.append(base * tier_boost * hedg_boost)

        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        selected = random.choices(candidates, weights=probs, k=batch_size)
        return [c[0] for c in selected]

    def _sample_fifo(
        self,
        batch_size: int,
        domain_id: Optional[int],
        min_quality: Optional[float],
        max_quality: Optional[float],
    ) -> List[dict]:
        """FIFO sampling (oldest first)."""
        candidates = self._collect_candidates(domain_id, min_quality, max_quality)
        return [c[0] for c in candidates[:batch_size]]

    def get_stats(self) -> Dict[str, float]:
        """Get bank statistics."""
        total = 0
        by_tier = {}
        quality_sum = 0.0
        hedg_sim_sum = 0.0
        hardness_sum = 0.0
        fallback_sum = 0.0

        for tier in range(self.num_tiers):
            tier_count = 0
            for domain in range(self.num_domains):
                tier_count += len(self.bank[tier][domain])
                total += len(self.bank[tier][domain])
            by_tier[f"tier_{tier}"] = tier_count

        for tier in range(self.num_tiers):
            for domain in range(self.num_domains):
                for entry in self.bank[tier][domain]:
                    quality_sum += entry["quality_score"]
                    hedg_sim_sum += float(entry.get("hedg_avg_similarity", 0.0))
                    hardness_sum += float(entry.get("hardness_proxy", 0.0))
                    fallback_sum += float(entry.get("hedg_fallback_rate", 0.0))

        avg_quality = quality_sum / max(total, 1)
        avg_hedg_sim = hedg_sim_sum / max(total, 1)
        avg_hardness = hardness_sum / max(total, 1)
        avg_fallback = fallback_sum / max(total, 1)

        return {
            "total_size": total,
            "max_size": self.max_size,
            "utilization": total / max(self.max_size, 1),
            "avg_quality": avg_quality,
            "avg_hedg_similarity": avg_hedg_sim,
            "avg_hardness_proxy": avg_hardness,
            "avg_hedg_fallback_rate": avg_fallback,
            **by_tier,
        }

    def reset(self):
        """Clear all entries."""
        for tier in range(self.num_tiers):
            for domain in range(self.num_domains):
                self.bank[tier][domain].clear()


class QualityAwareNegativeSampler:
    """
    Wraps existing negative sampling with quality-based filtering.
    """

    def __init__(
        self,
        hard_negative_bank: HardNegativeBank,
        min_quality: float = 0.0,
        max_quality: float = 0.4,
    ):
        """
        Args:
            hard_negative_bank: Bank for storing weak samples
            min_quality: Minimum quality for hard negatives
            max_quality: Maximum quality for hard negatives
        """
        self.hard_negative_bank = hard_negative_bank
        self.min_quality = min_quality
        self.max_quality = max_quality

    def add_candidate(
        self,
        subhg_data: dict,
        quality_score: float,
        domain_id: int,
    ):
        """Add a candidate to the hard negative bank if it qualifies."""
        if self.min_quality <= quality_score <= self.max_quality:
            self.hard_negative_bank.add(subhg_data, quality_score, domain_id)

    def sample_negatives(
        self,
        batch_size: int,
        domain_id: Optional[int] = None,
    ) -> List[dict]:
        """Sample hard negatives."""
        return self.hard_negative_bank.sample(
            batch_size,
            domain_id=domain_id,
            min_quality=self.min_quality,
            max_quality=self.max_quality,
        )
