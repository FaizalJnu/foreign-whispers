"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    # torchaudio 2.5+ removed the top-level AudioMetaData attribute that
    # pyannote.audio 3.x uses as a return-type annotation in io.py.  Patch
    # it back as a NamedTuple stub *before* the import so the annotation
    # evaluation doesn't raise AttributeError.  Only the attribute lookup
    # matters — this stub is never called at runtime.
    import torchaudio as _ta
    if not hasattr(_ta, "AudioMetaData"):
        import collections
        _ta.AudioMetaData = collections.namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError, AttributeError, Exception) as exc:  # noqa: BLE001
        logger.warning("pyannote.audio not importable — returning empty diarization. (%s)", exc)
        return []

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []

def assign_speakers(segments, diarization):
    result = []

    for seg in segments:
        seg_copy = seg.copy()

        if not diarization:
            seg_copy["speaker"] = "SPEAKER_00"
            result.append(seg_copy)
            continue

        best_speaker = "SPEAKER_00"
        best_overlap = 0

        for d in diarization:
            overlap = max(
                0,
                min(seg["end"], d["end_s"]) -
                max(seg["start"], d["start_s"])
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        seg_copy["speaker"] = best_speaker
        result.append(seg_copy)

    return result
