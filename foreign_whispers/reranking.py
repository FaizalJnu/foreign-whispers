# """Deterministic failure analysis and translation re-ranking stubs.

# The failure analysis function uses simple threshold rules derived from
# SegmentMetrics.  The translation re-ranking function is a **student assignment**
# — see the docstring for inputs, outputs, and implementation guidance.
# """

# import dataclasses
# import logging

# logger = logging.getLogger(__name__)


# @dataclasses.dataclass
# class TranslationCandidate:
#     """A candidate translation that fits a duration budget.

#     Attributes:
#         text: The translated text.
#         char_count: Number of characters in *text*.
#         brevity_rationale: Short explanation of what was shortened.
#     """
#     text: str
#     char_count: int
#     brevity_rationale: str = ""


# @dataclasses.dataclass
# class FailureAnalysis:
#     """Diagnostic summary of the dominant failure mode in a clip.

#     Attributes:
#         failure_category: One of "duration_overflow", "cumulative_drift",
#             "stretch_quality", or "ok".
#         likely_root_cause: One-sentence description.
#         suggested_change: Most impactful next action.
#     """
#     failure_category: str
#     likely_root_cause: str
#     suggested_change: str


# def analyze_failures(report: dict) -> FailureAnalysis:
#     """Classify the dominant failure mode from a clip evaluation report.

#     Pure heuristic — no LLM needed.  The thresholds below match the policy
#     bands defined in ``alignment.decide_action``.

#     Args:
#         report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
#             ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
#             ``total_cumulative_drift_s``, ``n_translation_retries``.

#     Returns:
#         A ``FailureAnalysis`` dataclass.
#     """
#     mean_err = report.get("mean_abs_duration_error_s", 0.0)
#     pct_severe = report.get("pct_severe_stretch", 0.0)
#     drift = abs(report.get("total_cumulative_drift_s", 0.0))
#     retries = report.get("n_translation_retries", 0)

#     if pct_severe > 20:
#         return FailureAnalysis(
#             failure_category="duration_overflow",
#             likely_root_cause=(
#                 f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
#                 "translated text is consistently too long for the available time window."
#             ),
#             suggested_change="Implement duration-aware translation re-ranking (P8).",
#         )

#     if drift > 3.0:
#         return FailureAnalysis(
#             failure_category="cumulative_drift",
#             likely_root_cause=(
#                 f"Total drift is {drift:.1f}s — small per-segment overflows "
#                 "accumulate because gaps between segments are not being reclaimed."
#             ),
#             suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
#         )

#     if mean_err > 0.8:
#         return FailureAnalysis(
#             failure_category="stretch_quality",
#             likely_root_cause=(
#                 f"Mean duration error is {mean_err:.2f}s — segments fit within "
#                 "stretch limits but the stretch distorts audio quality."
#             ),
#             suggested_change="Lower the mild_stretch ceiling or shorten translations.",
#         )

#     return FailureAnalysis(
#         failure_category="ok",
#         likely_root_cause="No dominant failure mode detected.",
#         suggested_change="Review individual outlier segments if any remain.",
#     )


# def get_shorter_translations(
#     source_text: str,
#     baseline_es: str,
#     target_duration_s: float,
#     context_prev: str = "",
#     context_next: str = "",
# ) -> list[TranslationCandidate]:
#     """Return shorter translation candidates that fit *target_duration_s*.

#     .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

#        This function is intentionally a **stub that returns an empty list**.
#        Your task is to implement a strategy that produces shorter
#        target-language translations when the baseline translation is too long
#        for the time budget.

#        **Inputs**

#        ============== ======== ==================================================
#        Parameter      Type     Description
#        ============== ======== ==================================================
#        source_text    str      Original source-language segment text
#        baseline_es    str      Baseline target-language translation (from argostranslate)
#        target_duration_s float Time budget in seconds for this segment
#        context_prev   str      Text of the preceding segment (for coherence)
#        context_next   str      Text of the following segment (for coherence)
#        ============== ======== ==================================================

#        **Outputs**

#        A list of ``TranslationCandidate`` objects, sorted shortest first.
#        Each candidate has:

#        - ``text``: the shortened target-language translation
#        - ``char_count``: ``len(text)``
#        - ``brevity_rationale``: short note on what was changed

#        **Duration heuristic**: target-language TTS produces ~15 characters/second
#        (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
#        ≈ 45 characters.

#        **Approaches to consider** (pick one or combine):

#        1. **Rule-based shortening** — strip filler words, use shorter synonyms
#           from a lookup table, contract common phrases
#           (e.g. "en este momento" → "ahora").
#        2. **Multiple translation backends** — call argostranslate with
#           paraphrased input, or use a second translation model, then pick
#           the shortest output that preserves meaning.
#        3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
#           condensed alternatives.  This was the previous approach but adds
#           latency, cost, and a runtime dependency.
#        4. **Hybrid** — rule-based first, fall back to LLM only for segments
#           that still exceed the budget.

#        **Evaluation criteria**: the caller selects the candidate whose
#        ``len(text) / 15.0`` is closest to ``target_duration_s``.

#     Returns:
#         Empty list (stub).  Implement to return ``TranslationCandidate`` items.
#     """

#     logger.info(
#         "get_shorter_translations called for %.1fs budget (%d chars baseline) — "
#         "returning empty list (student assignment stub).",
#         target_duration_s,
#         len(baseline_es),
#     )
#     return []
"""Deterministic failure analysis and translation re-ranking.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function implements a hybrid
strategy:

  1. Rule-based shortening  — contraction lookup + filler-word removal.
  2. Multiple backends      — re-translates the *source* through argostranslate
                              and (optionally) a MarianMT model, keeping the
                              shortest output that preserves rough meaning.
  3. LLM re-ranking         — falls back to the Groq API only when the
                              rule-based + backend passes still exceed the
                              character budget.

The caller selects the candidate whose ``len(text) / 15.0`` is closest to
``target_duration_s``.
"""

import dataclasses
import json
import logging
import re
import urllib.request
import urllib.error
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Duration heuristic
# ---------------------------------------------------------------------------

CHARS_PER_SECOND: float = 15.0  # Spanish TTS ~15 chars / second


def _char_budget(target_duration_s: float) -> int:
    """Return the maximum character count for *target_duration_s*."""
    return max(10, int(target_duration_s * CHARS_PER_SECOND))


# ---------------------------------------------------------------------------
# Rule-based shortening tables
# ---------------------------------------------------------------------------

# Ordered from longest match to shortest so greedy replacement works correctly.
SPANISH_CONTRACTIONS: list[tuple[str, str]] = [
    # verbose phrase → concise equivalent
    ("en este momento",         "ahora"),
    ("en el momento actual",    "ahora"),
    ("con el objetivo de",      "para"),
    ("con el fin de que",       "para que"),
    ("con el fin de",           "para"),
    ("a pesar de que",          "aunque"),
    ("a pesar de ello",         "aún así"),
    ("en el caso de que",       "si"),
    ("en caso de que",          "si"),
    ("debido a que",            "porque"),
    ("a causa de que",          "porque"),
    ("a causa de",              "por"),
    ("con respecto a",          "sobre"),
    ("en relación con",         "sobre"),
    ("en lo que respecta a",    "sobre"),
    ("con la finalidad de",     "para"),
    ("hacer referencia a",      "mencionar"),
    ("llevar a cabo",           "hacer"),
    ("tener en cuenta",         "considerar"),
    ("poner en práctica",       "aplicar"),
    ("dar a conocer",           "mostrar"),
    ("a lo largo de",           "durante"),
    ("en el transcurso de",     "durante"),
    ("en virtud de",            "por"),
    ("por lo tanto",            "así"),
    ("por consiguiente",        "así"),
    ("sin embargo",             "pero"),
    ("no obstante",             "pero"),
    ("a pesar de",              "pese a"),
    ("es decir",                "o sea"),
    ("en otras palabras",       "o sea"),
    ("de todas formas",         "igual"),
    ("de todas maneras",        "igual"),
    ("en definitiva",           "al fin"),
    ("en resumen",              "así"),
    ("en conclusión",           "así"),
    ("actualmente",             "hoy"),
    ("en la actualidad",        "hoy"),
    ("en estos momentos",       "ahora"),
    ("muy importante",          "crucial"),
    ("muy significativo",       "clave"),
    ("se puede observar",       "se ve"),
    ("se puede ver",            "se ve"),
    ("hay que señalar que",     "cabe señalar"),
    ("cabe destacar que",       "destaca"),
    ("es necesario",            "hay que"),
    ("es preciso",              "hay que"),
    ("es imprescindible",       "hay que"),
]

# Filler words / redundant adverbs safe to drop in isolation
FILLER_WORDS: list[str] = [
    r"\bverdaderamente\b",
    r"\bprácticamente\b",
    r"\bfundamentalmente\b",
    r"\bbásicamente\b",
    r"\bciertamente\b",
    r"\bindudablemente\b",
    r"\binnegablemente\b",
    r"\bobviamente\b",
    r"\bevidentmente\b",
    r"\btotalmente\b",
    r"\bcompletamente\b",
    r"\babsolutamente\b",
    r"\bgeneralmente\b",
    r"\bnormalmente\b",
    r"\bhabitualmente\b",
    r"\bparticularmente\b",
    r"\bespecialmente\b",
    r"\bconsiderablemente\b",
    r"\bsignificativamente\b",
    r"\bprofundamente\b",
    r"\bgrandemente\b",
    r"\bampliamente\b",
]

_FILLER_RE = re.compile("|".join(FILLER_WORDS), re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r" {2,}")


def _apply_contractions(text: str) -> str:
    """Apply SPANISH_CONTRACTIONS lookup (case-insensitive, longest first)."""
    lower = text.lower()
    result = text
    offset = 0  # character shift caused by replacements so far
    for phrase, replacement in SPANISH_CONTRACTIONS:
        idx = lower.find(phrase)
        while idx != -1:
            # Preserve capitalisation of the first character
            rep = replacement[0].upper() + replacement[1:] if result[idx].isupper() else replacement
            result = result[:idx] + rep + result[idx + len(phrase):]
            lower  = result.lower()
            idx    = lower.find(phrase, idx + len(rep))
    return result


def _strip_fillers(text: str) -> str:
    """Remove standalone filler adverbs and collapse extra whitespace."""
    cleaned = _FILLER_RE.sub("", text)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    # Remove dangling commas left behind by filler removal
    cleaned = re.sub(r"\s*,\s*,", ",", cleaned)
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    return cleaned


def _rule_based_shorten(text: str) -> str:
    """Apply contraction + filler-strip pipeline and return the result."""
    return _strip_fillers(_apply_contractions(text))


# ---------------------------------------------------------------------------
# Multi-backend translation
# ---------------------------------------------------------------------------

def _argostranslate_translate(text: str, from_code: str = "en", to_code: str = "es") -> Optional[str]:
    """Attempt to translate *text* via argostranslate; return None on failure."""
    try:
        from argostranslate import package, translate  # type: ignore

        installed = translate.get_installed_languages()
        src_lang = next((l for l in installed if l.code == from_code), None)
        tgt_lang = next((l for l in installed if l.code == to_code), None)
        if src_lang is None or tgt_lang is None:
            logger.debug("argostranslate: language pair %s→%s not installed.", from_code, to_code)
            return None
        translation = src_lang.get_translation(tgt_lang)
        return translation.translate(text) if translation else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("argostranslate translation failed: %s", exc)
        return None


def _marianmt_translate(text: str, model_name: str = "Helsinki-NLP/opus-mt-en-es") -> Optional[str]:
    """Attempt to translate *text* via a local MarianMT model; return None on failure."""
    try:
        from transformers import MarianMTModel, MarianTokenizer  # type: ignore

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model     = MarianMTModel.from_pretrained(model_name)
        tokens    = tokenizer([text], return_tensors="pt", padding=True)
        translated = model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("MarianMT translation failed: %s", exc)
        return None


def _multi_backend_candidates(
    source_text: str,
    baseline_es: str,
    char_budget: int,
) -> list["TranslationCandidate"]:
    """Run source text through available translation backends.

    Returns candidates shorter than *baseline_es* (or within budget), sorted
    shortest first.
    """
    candidates: list[TranslationCandidate] = []

    backends = [
        ("argostranslate", lambda t: _argostranslate_translate(t)),
        ("marianmt",       lambda t: _marianmt_translate(t)),
    ]

    for name, fn in backends:
        result = fn(source_text)
        if result and result.strip() and result.strip() != baseline_es.strip():
            rule_shortened = _rule_based_shorten(result)
            candidates.append(TranslationCandidate(
                text=rule_shortened,
                char_count=len(rule_shortened),
                brevity_rationale=f"retranslated via {name} + rule pass",
            ))
            # Also keep the raw backend output if different
            if rule_shortened != result.strip():
                candidates.append(TranslationCandidate(
                    text=result.strip(),
                    char_count=len(result.strip()),
                    brevity_rationale=f"retranslated via {name}",
                ))

    return sorted(candidates, key=lambda c: c.char_count)


# ---------------------------------------------------------------------------
# LLM re-ranking (GROQ API — fallback only)
# ---------------------------------------------------------------------------


def _llm_shorten(
    source_text: str,
    baseline_es: str,
    char_budget: int,
    context_prev: str,
    context_next: str,
    n_candidates: int = 3,
) -> list["TranslationCandidate"]:
    """Call the Groq API to generate *n_candidates* shorter translations.

    Returns an empty list if the API call fails or the key is unavailable.
    The API key is read from the ``GROQ_API_KEY`` environment variable.
    """
    import os  # local import — only needed in this branch

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.debug("GROQ_API_KEY not set; skipping LLM re-ranking.")
        return []

    system_prompt = (
        "You are a professional Spanish translator specialising in concise subtitles. "
        "Always respond with valid JSON only — no markdown fences, no commentary."
    )

    user_prompt = (
        f"The following Spanish translation is too long for its TTS time slot.\n\n"
        f"Source (English): {source_text}\n"
        f"Baseline Spanish (too long, {len(baseline_es)} chars): {baseline_es}\n"
        f"Character budget: {char_budget} chars (≈{char_budget / CHARS_PER_SECOND:.1f}s at 15 chars/s)\n"
        + (f"Previous segment: {context_prev}\n" if context_prev else "")
        + (f"Next segment: {context_next}\n" if context_next else "")
        + f"\nGenerate {n_candidates} alternative Spanish translations, each strictly under "
        f"{char_budget} characters, preserving the core meaning. "
        f"Return a JSON array sorted shortest first:\n"
        f'[{{"text": "...", "rationale": "one-line note"}}]'
    )

    payload = json.dumps({
        "model":      _GROQ_MODEL,
        "max_tokens": 1000,
        "system":     system_prompt,
        "messages":   [{"role": "user", "content": user_prompt}],
    }).encode()

    req = urllib.request.Request(
        _ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        logger.warning("LLM re-ranking HTTP error: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM re-ranking unexpected error: %s", exc)
        return []

    # Extract text from the first content block
    try:
        raw_text = body["content"][0]["text"].strip()
        # Strip accidental markdown fences
        raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text)
        items = json.loads(raw_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM re-ranking JSON parse failed: %s", exc)
        return []

    candidates: list[TranslationCandidate] = []
    for item in items:
        text      = str(item.get("text", "")).strip()
        rationale = str(item.get("rationale", "llm condensed"))
        if text:
            candidates.append(TranslationCandidate(
                text=text,
                char_count=len(text),
                brevity_rationale=rationale,
            ))

    return sorted(candidates, key=lambda c: c.char_count)


# ---------------------------------------------------------------------------
# Dataclasses (unchanged public API)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


# ---------------------------------------------------------------------------
# Failure analysis (unchanged)
# ---------------------------------------------------------------------------

def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err   = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift      = abs(report.get("total_cumulative_drift_s", 0.0))
    retries    = report.get("n_translation_retries", 0)  # noqa: F841 (kept for future use)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4× stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


# ---------------------------------------------------------------------------
# Main entry point: hybrid re-ranking
# ---------------------------------------------------------------------------

def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    Implements a three-tier hybrid strategy:

    1. **Rule-based** — apply Spanish contraction lookup and filler-word
       stripping.  Zero latency, zero cost.
    2. **Multi-backend** — re-translate *source_text* through every available
       backend (argostranslate, MarianMT).  The shorter raw output and its
       rule-processed variant are both kept.
    3. **LLM fallback** — only when no candidate from tiers 1–2 fits within
       the budget, call the Groq API to generate condensed alternatives.

    Args:
        source_text: Original source-language (English) segment text.
        baseline_es: Baseline Spanish translation from argostranslate.
        target_duration_s: Time budget in seconds for this segment.
        context_prev: Text of the preceding segment (for LLM coherence).
        context_next: Text of the following segment (for LLM coherence).

    Returns:
        List of ``TranslationCandidate`` objects, sorted shortest first.
        The baseline itself is appended as the last (longest) fallback so
        the caller always receives at least one candidate.
    """
    char_budget = _char_budget(target_duration_s)
    baseline_chars = len(baseline_es)

    logger.info(
        "get_shorter_translations: budget=%d chars (%.1fs), baseline=%d chars.",
        char_budget, target_duration_s, baseline_chars,
    )

    # Fast path: baseline already fits.
    if baseline_chars <= char_budget:
        logger.debug("Baseline fits budget — returning baseline only.")
        return [TranslationCandidate(
            text=baseline_es,
            char_count=baseline_chars,
            brevity_rationale="baseline fits budget",
        )]

    candidates: list[TranslationCandidate] = []

    # ------------------------------------------------------------------
    # Tier 1: rule-based shortening of the baseline
    # ------------------------------------------------------------------
    rule_text = _rule_based_shorten(baseline_es)
    if rule_text != baseline_es:
        candidates.append(TranslationCandidate(
            text=rule_text,
            char_count=len(rule_text),
            brevity_rationale="contracted phrases + filler removal",
        ))
        logger.debug("Rule pass: %d → %d chars.", baseline_chars, len(rule_text))

    # ------------------------------------------------------------------
    # Tier 2: multi-backend re-translation of the source
    # ------------------------------------------------------------------
    backend_candidates = _multi_backend_candidates(source_text, baseline_es, char_budget)
    candidates.extend(backend_candidates)
    logger.debug("Backend pass produced %d candidate(s).", len(backend_candidates))

    # ------------------------------------------------------------------
    # Tier 3: LLM fallback — only if nothing fits the budget yet
    # ------------------------------------------------------------------
    fits_budget = any(c.char_count <= char_budget for c in candidates)
    if not fits_budget:
        logger.info(
            "No candidate fits budget (%d chars) after rule+backend passes — "
            "invoking LLM re-ranking.",
            char_budget,
        )
        llm_candidates = _llm_shorten(
            source_text=source_text,
            baseline_es=baseline_es,
            char_budget=char_budget,
            context_prev=context_prev,
            context_next=context_next,
        )
        candidates.extend(llm_candidates)
        logger.debug("LLM pass produced %d candidate(s).", len(llm_candidates))
    else:
        logger.debug("Budget satisfied without LLM — skipping API call.")

    # ------------------------------------------------------------------
    # Always append the baseline as the last-resort fallback
    # ------------------------------------------------------------------
    candidates.append(TranslationCandidate(
        text=baseline_es,
        char_count=baseline_chars,
        brevity_rationale="baseline (fallback)",
    ))

    # Deduplicate (preserve order, prefer first occurrence which is shortest)
    seen: set[str] = set()
    unique: list[TranslationCandidate] = []
    for c in candidates:
        if c.text not in seen:
            seen.add(c.text)
            unique.append(c)

    # Sort shortest first so the caller can pick the best fit easily
    return sorted(unique, key=lambda c: c.char_count)