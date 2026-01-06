"""
Utility functions for memory system.

Core utilities for scoring, similarity calculation, and temporal processing.
These functions are used throughout the engine for memory operations.

Adapted from hindsight-api for memory-api.
"""

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_wrapper import LLMProvider

logger = logging.getLogger(__name__)


# ============================================
# Vector Similarity
# ============================================

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score between -1 and 1 (typically 0 to 1 for embeddings)
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same dimension: {len(vec1)} vs {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Distance (lower = more similar)
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same dimension: {len(vec1)} vs {len(vec2)}")

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


# ============================================
# Temporal Calculations
# ============================================

def calculate_recency_weight(days_since: float, half_life_days: float = 365.0) -> float:
    """
    Calculate recency weight using logarithmic decay.

    This provides much better differentiation over long time periods compared to
    exponential decay. Uses a log-based decay where the half-life parameter controls
    when memories reach 50% weight.

    Examples:
        - Today (0 days): 1.0
        - 1 year (365 days): ~0.5 (with default half_life=365)
        - 2 years (730 days): ~0.33
        - 5 years (1825 days): ~0.17
        - 10 years (3650 days): ~0.09

    This ensures that 2-year-old and 5-year-old memories have meaningfully
    different weights, unlike exponential decay which makes them both ~0.

    Args:
        days_since: Number of days since the memory was created
        half_life_days: Number of days for weight to reach 0.5 (default: 1 year)

    Returns:
        Weight between 0 and 1
    """
    if days_since < 0:
        days_since = 0.0
    
    # Logarithmic decay: 1 / (1 + log(1 + days_since/half_life))
    # This decays much slower than exponential, giving better long-term differentiation
    normalized_age = days_since / half_life_days
    return 1.0 / (1.0 + math.log1p(normalized_age))


def calculate_frequency_weight(access_count: int, max_boost: float = 2.0) -> float:
    """
    Calculate frequency weight based on access count.

    Frequently accessed memories are weighted higher.
    Uses logarithmic scaling to avoid over-weighting.

    Args:
        access_count: Number of times the memory was accessed
        max_boost: Maximum multiplier for frequently accessed memories

    Returns:
        Weight between 1.0 and max_boost
    """
    if access_count <= 0:
        return 1.0

    # Logarithmic scaling: log(access_count + 1) / log(10)
    # This gives: 0 accesses = 1.0, 9 accesses ~= 1.5, 99 accesses ~= 2.0
    normalized = math.log(access_count + 1) / math.log(10)
    return 1.0 + min(normalized, max_boost - 1.0)


def calculate_temporal_anchor(occurred_start: datetime, occurred_end: datetime) -> datetime:
    """
    Calculate a single temporal anchor point from a temporal range.

    Used for spreading activation - we need a single representative date
    to calculate temporal proximity between facts. This simplifies the
    range-to-range distance problem.

    Strategy: Use midpoint of the range for balanced representation.

    Args:
        occurred_start: Start of temporal range
        occurred_end: End of temporal range

    Returns:
        Single datetime representing the temporal anchor (midpoint)

    Examples:
        - Point event (July 14): start=July 14, end=July 14 → anchor=July 14
        - Month range (February): start=Feb 1, end=Feb 28 → anchor=Feb 14
        - Year range (2023): start=Jan 1, end=Dec 31 → anchor=July 1
    """
    # Calculate midpoint
    time_delta = occurred_end - occurred_start
    midpoint = occurred_start + (time_delta / 2)
    return midpoint


def calculate_temporal_proximity(
    anchor_a: datetime, 
    anchor_b: datetime, 
    half_life_days: float = 30.0
) -> float:
    """
    Calculate temporal proximity between two temporal anchors.

    Used for spreading activation to determine how "close" two facts are
    in time. Uses logarithmic decay so that temporal similarity doesn't
    drop off too quickly.

    Args:
        anchor_a: Temporal anchor of first fact
        anchor_b: Temporal anchor of second fact
        half_life_days: Number of days for proximity to reach 0.5
                       (default: 30 days = 1 month)

    Returns:
        Proximity score in [0, 1] where:
        - 1.0 = same day
        - 0.5 = ~half_life days apart
        - 0.0 = very distant in time

    Examples:
        - Same day: 1.0
        - 1 week apart (half_life=30): ~0.7
        - 1 month apart (half_life=30): ~0.5
        - 1 year apart (half_life=30): ~0.2
    """
    days_apart = abs((anchor_a - anchor_b).days)

    if days_apart == 0:
        return 1.0

    # Logarithmic decay: 1 / (1 + log(1 + days_apart/half_life))
    # Similar to calculate_recency_weight but for proximity between events
    normalized_distance = days_apart / half_life_days
    proximity = 1.0 / (1.0 + math.log1p(normalized_distance))

    return proximity


def days_between(date1: datetime, date2: datetime) -> float:
    """
    Calculate the number of days between two dates.
    
    Args:
        date1: First date
        date2: Second date
        
    Returns:
        Number of days (can be fractional)
    """
    delta = date2 - date1
    return delta.total_seconds() / 86400.0  # 86400 seconds in a day


# ============================================
# Scoring Utilities
# ============================================

def normalize_score(score: float, min_score: float, max_score: float) -> float:
    """
    Normalize a score to [0, 1] range.
    
    Args:
        score: Raw score to normalize
        min_score: Minimum possible score
        max_score: Maximum possible score
        
    Returns:
        Normalized score in [0, 1]
    """
    if max_score == min_score:
        return 0.5
    
    normalized = (score - min_score) / (max_score - min_score)
    return max(0.0, min(1.0, normalized))


def sigmoid(x: float, k: float = 1.0) -> float:
    """
    Sigmoid function for smooth scoring transitions.
    
    Args:
        x: Input value
        k: Steepness parameter (higher = steeper curve)
        
    Returns:
        Value in (0, 1)
    """
    try:
        return 1.0 / (1.0 + math.exp(-k * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def combine_scores(
    scores: list[float], 
    weights: list[float] | None = None
) -> float:
    """
    Combine multiple scores using weighted average.
    
    Args:
        scores: List of scores to combine
        weights: Optional weights for each score (defaults to equal weights)
        
    Returns:
        Combined score
    """
    if not scores:
        return 0.0
    
    if weights is None:
        weights = [1.0] * len(scores)
    
    if len(scores) != len(weights):
        raise ValueError("Scores and weights must have same length")
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / total_weight


# ============================================
# Text Utilities
# ============================================

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean text for processing.
    
    - Strips whitespace
    - Normalizes multiple spaces to single space
    - Removes control characters
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    import re
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


# ============================================
# Batch Processing Utilities
# ============================================

def batch_items(items: list, batch_size: int) -> list[list]:
    """
    Split items into batches.
    
    Args:
        items: List of items to batch
        batch_size: Maximum size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


async def process_in_batches(items: list, processor, batch_size: int = 100):
    """
    Process items in batches using an async processor.
    
    Args:
        items: Items to process
        processor: Async function that takes a batch and returns results
        batch_size: Size of each batch
        
    Returns:
        Combined results from all batches
    """
    results = []
    for batch in batch_items(items, batch_size):
        batch_results = await processor(batch)
        results.extend(batch_results)
    return results


# ============================================
# Hash Utilities
# ============================================

def compute_content_hash(content: str) -> str:
    """
    Compute a hash of content for deduplication.
    
    Args:
        content: Text content to hash
        
    Returns:
        SHA-256 hash as hex string
    """
    import hashlib
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def generate_chunk_id(bank_id: str, document_id: str, chunk_index: int) -> str:
    """
    Generate a unique chunk ID.
    
    Args:
        bank_id: Memory bank ID
        document_id: Document ID
        chunk_index: Index of the chunk
        
    Returns:
        Unique chunk ID
    """
    return f"{bank_id}_{document_id}_{chunk_index}"
