"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.
- ``global_align_dp`` — LP-based global optimizer that allocates silence
  budgets to the segments that need them most before scheduling.

No external dependencies beyond numpy and scipy.
"""
import dataclasses
import re
import unicodedata
from enum import Enum
from pathlib import Path
import joblib

try:
    PROJECT_ROOT = Path(__file__).parent.parent
    DURATION_MODEL = joblib.load(PROJECT_ROOT / "pipeline_data" / "duration_model.pkl")
except FileNotFoundError:
    DURATION_MODEL = None
    print("Warning: duration_model.pkl not found. Falling back to heuristic.")



def _count_syllables(text: str) -> int:
    """Count syllables in target-language text via vowel-cluster counting.

    Designed for Romance languages (Spanish, French, Italian, Portuguese).
    Strips accents then counts contiguous vowel runs. Each run = one syllable.
    Returns at least 1 for any non-empty text so the rate never divides by zero.
    """
    # Normalise: decompose accented chars, keep only ASCII letters + spaces
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    clusters = re.findall(r"[aeiou]+", ascii_text)
    return max(1, len(clusters))


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5 syllables/second for Romance languages) and derive three key numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        predicted_tts_s: Estimated TTS duration (syllables / 4.5).
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30% longer than the available window.
        overflow_s: How many seconds the target-language audio exceeds the
            window (zero when it fits).
    """
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if DURATION_MODEL:
            # ML INFERENCE PATH
            chars = len(self.translated_text)
            words = len(self.translated_text.split())
            syllables = _count_syllables(self.translated_text)
            punctuation = len(re.findall(r"[,.;:!?]", self.translated_text))
            
            features = [[chars, words, syllables, punctuation]]
            # Predict returns an array, grab the first element
            self.predicted_tts_s = float(DURATION_MODEL.predict(features)[0])
            
            # Prevent negative duration predictions from the linear model
            self.predicted_tts_s = max(0.5, self.predicted_tts_s) 
            
        else:
            # FALLBACK HEURISTIC PATH (If model fails to load)
            syllables = _count_syllables(self.translated_text)
            self.predicted_tts_s = syllables / 4.5

        # The rest remains the same
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy.

    Each segment gets exactly one action based on its ``predicted_stretch``:

    - ``ACCEPT`` — fits within 10% of the original duration, no change needed.
    - ``MILD_STRETCH`` — 10–40% over; apply pyrubberband time-stretch.
    - ``GAP_SHIFT`` — 40–80% over but adjacent silence can absorb the overflow.
    - ``REQUEST_SHORTER`` — 80–150% over; needs a shorter translation (P8).
    - ``FAIL`` — >150% over; no fix available, log and fall back to silence.
    """
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline.

    Produced by ``global_align``.  The ``scheduled_start`` and
    ``scheduled_end`` incorporate cumulative drift from earlier gap shifts,
    so they may differ from the original Whisper timestamps.

    Attributes:
        index: Segment position (matches ``SegmentMetrics.index``).
        original_start: Whisper start time (seconds).
        original_end: Whisper end time (seconds).
        scheduled_start: Start time after global alignment (seconds).
        scheduled_end: End time after global alignment (seconds).
        text: Target-language translated text for this segment.
        action: The ``AlignAction`` chosen by ``decide_action``.
        gap_shift_s: Seconds borrowed from adjacent silence (0.0 if none).
        stretch_factor: Speed factor for pyrubberband (1.0 = no stretch).
    """
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment.

    Maps the predicted stretch factor to one of five actions using fixed
    thresholds.  ``GAP_SHIFT`` additionally requires that enough silence
    follows the segment to absorb the overflow.

    Thresholds::

        predicted_stretch   Action            Condition
        ─────────────────   ────────────────  ─────────────────────────
        <= 1.1              ACCEPT            fits naturally
        1.1 – 1.4          MILD_STRETCH      pyrubberband safe range
        1.4 – 1.8          GAP_SHIFT         only if gap >= overflow
        1.8 – 2.5          REQUEST_SHORTER   needs shorter translation
        > 2.5              FAIL              unfixable

    Args:
        m: Timing metrics for one segment.
        available_gap_s: Silence duration (seconds) after this segment,
            from VAD.  Defaults to 0.0 (no gap available).

    Returns:
        The ``AlignAction`` to apply.
    """
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics.

    Zips the ``"segments"`` lists from both transcripts positionally
    (segment 0 ↔ segment 0, etc.) and builds a ``SegmentMetrics`` for each
    pair.  The source segment provides the time window; the target segment
    provides the text whose TTS duration we need to predict.

    Args:
        en_transcript: Source-language Whisper output dict with
            ``{"segments": [{"start", "end", "text"}, ...]}``.
        es_transcript: Target-language translation dict with the same structure.

    Returns:
        List of ``SegmentMetrics``, one per paired segment.  If the transcripts
        have different lengths, the shorter one determines the output length.
    """
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        metrics.append(SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
        ))
    return metrics


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments.

    Segments are timed independently by ``decide_action`` (P7), but they are
    sequential — if segment 5 borrows 0.3s from a silence gap, every segment
    after it shifts by 0.3s.  This function tracks that cumulative drift.

    Algorithm (single pass, O(n)):

    1. For each segment, call ``decide_action(m, available_gap_s)`` where
       *available_gap_s* comes from VAD silence regions after this segment.
    2. Based on the action:

       - ``GAP_SHIFT`` — the segment expands into the silence after it
         (``gap_shift = overflow_s``).
       - ``MILD_STRETCH`` — time-stretch capped at *max_stretch* (default 1.4x).
       - ``ACCEPT``, ``REQUEST_SHORTER``, ``FAIL`` — no modification.

    3. Schedule the segment with cumulative drift applied::

           scheduled_start = original_start + cumulative_drift
           scheduled_end   = scheduled_start + original_duration + gap_shift

    4. Every ``gap_shift`` adds to *cumulative_drift*, pushing all subsequent
       segments forward.

    Limitations:

    - **Greedy** — never looks ahead.  If segment 10 has a huge overflow and
      segment 9 has a large silence gap, it will not save that gap for
      segment 10.
    - **No backtracking** — once a decision is made, it is final.
    - A dynamic-programming or constraint-solver approach would produce
      better schedules, but this is the baseline to start from.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable (gap_shift disabled).
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        action    = decide_action(m, available_gap_s=_silence_after(m.source_end))
        gap_shift = 0.0
        stretch   = 1.0

        if action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)
        # ACCEPT, REQUEST_SHORTER, FAIL → stretch stays at 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned


def _estimate_duration(metrics: list[SegmentMetrics]) -> float:
    """Compute total duration based on the new ML/Syllable predicted TTS durations.

    This provides a target duration independent of VAD-based alignment, useful
    as a fallback or reference.

    Args:
        metrics: Per-segment timing metrics.

    Returns:
        Total duration in seconds: sum over segments of predicted_tts_s.
    """
    return sum(m.predicted_tts_s for m in metrics)


import numpy as np
from scipy.optimize import linprog


def global_align_dp(
    metrics: list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch: float = 1.4,
) -> list[AlignedSegment]:
    """Global alignment via linear programming over silence-gap allocations.

    Unlike the greedy pass, which awards silence to the first segment that
    needs it, this optimizer treats gap allocation as a global resource
    problem: it finds the assignment of silence budgets to segments that
    minimises *total* overflow across the whole clip.

    Formulation
    -----------
    Decision variables (one per segment)::

        g_i  ∈ [0, gap_capacity_i]   — seconds of silence allocated to segment i

    Objective — minimise total weighted overflow after allocation::

        minimise  Σ_i  w_i * max(0, overflow_i - g_i)

    Because ``linprog`` requires a linear objective, we introduce slack
    variables ``s_i ≥ 0`` and rewrite::

        minimise  Σ_i  w_i * s_i
        subject to:
            s_i  ≥  overflow_i - g_i        (slack absorbs remaining overflow)
            s_i  ≥  0
            0  ≤  g_i  ≤  gap_capacity_i    (can't borrow more than available)
            Σ_i  g_i  ≤  total_silence      (global silence budget)

    Weights ``w_i`` are set to ``predicted_stretch_i`` so that segments
    already far over budget are prioritised over mildly overflowing ones.

    After solving, each segment is scheduled greedily left-to-right with its
    allocated gap, producing a non-overlapping timeline.  The ``AlignAction``
    is derived from the *post-allocation* stretch factor, which is the key
    improvement over the greedy baseline: segments that received gap budget
    may now fall into ``MILD_STRETCH`` or ``ACCEPT`` instead of
    ``REQUEST_SHORTER``.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable.
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    n = len(metrics)
    if n == 0:
        return []

    # ------------------------------------------------------------------
    # 1. Build per-segment gap capacities from VAD silence regions.
    #    A silence region is eligible for segment i if it starts within
    #    0.1 s of that segment's end (matching the greedy heuristic).
    # ------------------------------------------------------------------
    def _gap_capacity(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    gap_caps      = np.array([_gap_capacity(m.source_end) for m in metrics])
    overflows     = np.array([m.overflow_s                for m in metrics])
    weights       = np.array([m.predicted_stretch          for m in metrics])
    total_silence = gap_caps.sum()

    # ------------------------------------------------------------------
    # 2. Formulate the LP.
    #
    #    Variable layout:  x = [g_0 … g_{n-1}, s_0 … s_{n-1}]
    #                           (gap allocs)    (slack / residual overflow)
    #
    #    Objective:  minimise  0·g  +  w·s
    # ------------------------------------------------------------------
    c = np.concatenate([np.zeros(n), weights])

    # Inequality constraints  A_ub @ x <= b_ub
    # (a)  s_i >= overflow_i - g_i   →   -g_i - s_i <= -overflow_i
    A_slack = np.zeros((n, 2 * n))
    for i in range(n):
        A_slack[i, i]     = -1.0   # -g_i
        A_slack[i, n + i] = -1.0   # -s_i
    b_slack = -overflows

    # (b)  Σ g_i <= total_silence
    A_budget = np.zeros((1, 2 * n))
    A_budget[0, :n] = 1.0
    b_budget = np.array([total_silence])

    A_ub = np.vstack([A_slack, A_budget])
    b_ub = np.concatenate([b_slack, b_budget])

    # Bounds:  g_i ∈ [0, gap_caps_i],  s_i ∈ [0, ∞)
    bounds = (
        [(0.0, float(cap)) for cap in gap_caps]
        + [(0.0, None)      for _   in range(n)]
    )

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not res.success:
        # Solver failed — fall back to the greedy result rather than crashing.
        print(f"⚠️  global_align_dp solver warning: {res.message}. "
              "Falling back to greedy.")
        return global_align(metrics, silence_regions, max_stretch)

    allocated_gaps = res.x[:n]   # g_i values from the LP solution

    # ------------------------------------------------------------------
    # 3. Schedule segments left-to-right using the LP-allocated gaps.
    #    Cumulative drift accumulates just as in global_align.
    # ------------------------------------------------------------------
    aligned: list[AlignedSegment] = []
    cumulative_drift = 0.0

    for i, m in enumerate(metrics):
        gap_shift = float(allocated_gaps[i])

        # Post-allocation stretch: how much does the window need to cover?
        available_s  = m.source_duration_s + gap_shift
        post_stretch = m.predicted_tts_s / available_s if available_s > 0 else 1.0

        # Derive action from post-allocation stretch — key difference from greedy,
        # which derives action before any gap is assigned.
        if post_stretch <= 1.1:
            action  = AlignAction.ACCEPT
            stretch = 1.0
        elif post_stretch <= 1.4:
            action  = AlignAction.MILD_STRETCH
            stretch = min(post_stretch, max_stretch)
        elif post_stretch <= 1.8 and gap_shift > 0:
            # Gap was partially allocated but didn't fully cover overflow.
            action  = AlignAction.GAP_SHIFT
            stretch = 1.0
        elif post_stretch <= 2.5:
            action  = AlignAction.REQUEST_SHORTER
            stretch = 1.0
        else:
            action  = AlignAction.FAIL
            stretch = 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned