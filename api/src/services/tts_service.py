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