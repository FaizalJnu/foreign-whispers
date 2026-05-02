"""Clip-level alignment quality metrics and multi-dimensional scorecard.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment.
"""
import statistics as _stats
import numpy as np
import re
import unicodedata
from scipy.spatial.distance import cosine

# ---------------------------------------------------------------------------
# Optional ML dependencies for the scorecard
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    # Load the multilingual model globally for semantic scoring
    semantic_embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except ImportError:
    semantic_embedder = None
    print("Warning: SentenceTransformer not loaded. Semantic score will default to 0.0.")

try:
    import jiwer
except ImportError:
    jiwer = None
    print("Warning: 'jiwer' not installed. Intelligibility (WER) will default to 1.0.")

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)

# ---------------------------------------------------------------------------
# Baseline Evaluation (Task 3)
# ---------------------------------------------------------------------------

def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }

# ---------------------------------------------------------------------------
# Multi-Dimensional Scorecard (Task 4)
# ---------------------------------------------------------------------------

def _count_syllables(text: str) -> int:
    """Helper to count syllables for naturalness scoring."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    clusters = re.findall(r"[aeiouáéíóúü]+", ascii_text)
    return max(1, len(clusters))

def dubbing_scorecard(
    metrics: list[SegmentMetrics], 
    aligned_segments: list[AlignedSegment], 
    align_report: dict,
    stt_transcripts: dict = None
) -> dict:
    """
    Computes a [0, 1] normalized scorecard across 4 quality dimensions.
    
    Args:
        metrics: List of SegmentMetrics.
        aligned_segments: List of AlignedSegment outputs from the optimizer.
        align_report: The baseline evaluation dict containing drift/stretch errors.
        stt_transcripts: Optional dict mapping segment index to a round-trip STT string.
    """
    scores = {}

    # 1. Timing Accuracy [0, 1]
    # Penalizes mean duration error, severe stretches, and drift.
    mean_err = align_report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = align_report.get("pct_severe_stretch", 0.0) / 100.0
    drift = align_report.get("total_cumulative_drift_s", 0.0)
    
    timing_penalty = (min(1.0, mean_err / 2.0) + min(1.0, pct_severe / 0.2) + min(1.0, drift / 5.0)) / 3.0
    scores["timing"] = round(max(0.0, 1.0 - timing_penalty), 3)

    # 2. Semantic Fidelity [0, 1]
    # Cosine similarity between original English and target Spanish
    if semantic_embedder and metrics:
        similarities = []
        for m in metrics:
            emb_en = semantic_embedder.encode(m.source_text)
            emb_es = semantic_embedder.encode(m.translated_text)
            sim = 1.0 - cosine(emb_en, emb_es)
            similarities.append(sim)
        scores["semantic"] = round(max(0.0, float(np.mean(similarities))), 3)
    else:
        scores["semantic"] = 0.0

    # 3. Naturalness (Speaking Rate Variance) [0, 1]
    # Inconsistent speaking rates (fast -> slow -> fast) sound robotic.
    if aligned_segments:
        rates = []
        for seg in aligned_segments:
            dur = seg.scheduled_end - seg.scheduled_start
            if dur > 0:
                syllables = _count_syllables(seg.text)
                rates.append(syllables / dur)
        
        rate_std = np.std(rates)
        scores["naturalness"] = round(max(0.0, 1.0 - min(1.0, rate_std / 4.0)), 3)
    else:
        scores["naturalness"] = 0.0

    # 4. Intelligibility (Round-Trip STT Word Error Rate) [0, 1]
    if jiwer and stt_transcripts and aligned_segments:
        wers = []
        for seg in aligned_segments:
            reference = seg.text
            hypothesis = stt_transcripts.get(seg.index, reference) 
            try:
                error_rate = jiwer.wer(reference, hypothesis)
                wers.append(error_rate)
            except ValueError:
                wers.append(1.0)
                
        mean_wer = np.mean(wers)
        scores["intelligibility"] = round(max(0.0, 1.0 - min(1.0, mean_wer / 0.5)), 3)
    else:
        scores["intelligibility"] = 1.0 

    # Calculate overall aggregate score
    scores["overall"] = round(sum(scores.values()) / len(scores), 3)
    
    return scores