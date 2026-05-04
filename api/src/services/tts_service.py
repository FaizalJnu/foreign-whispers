"""HTTP-agnostic service wrapping TTS engine functions."""

import json
import pathlib
from pathlib import Path
from typing import Any

from api.src.services.tts_engine import text_file_to_speech as tts_text_file_to_speech

class TTSService:
    """Thin wrapper around the TTS pipeline.

    Accepts *ui_dir* and a pre-loaded *tts_engine* via constructor injection.
    """

    def __init__(self, ui_dir: Path, tts_engine: Any) -> None:
        self.ui_dir = ui_dir
        self.tts_engine = tts_engine

    def text_file_to_speech(self, source_path: str, output_path: str,
                            *, alignment: bool | None = None,
                            speaker_wav: str | None = None) -> None:
        # Use the engine function imported at module level
        tts_text_file_to_speech(
            source_path,
            output_path,
            self.tts_engine,
            alignment=alignment,
            speaker_wav=speaker_wav,
        )

    def compute_alignment(
        self,
        en_transcript: dict,
        es_transcript: dict,
        silence_regions: list,
        max_stretch: float = 1.4,
    ) -> list:
        """Compute aligned segments for eval/preview without running TTS.

        Returns a list of AlignedSegment objects from global_align.
        Falls back to an empty list if the alignment library is unavailable.
        """
        try:
            from foreign_whispers.alignment import compute_segment_metrics, global_align
        except ImportError:
            return []
        metrics = compute_segment_metrics(en_transcript, es_transcript)
        return global_align(metrics, silence_regions, max_stretch=max_stretch)