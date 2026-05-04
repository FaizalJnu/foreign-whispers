"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService
from foreign_whispers.voice_resolution import resolve_speaker_wav

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    speaker_wav: str = Query(None, description="Reference voice WAV path (e.g. 'es/default.wav')"),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    # Capture the raw query-param value before any fallback is applied.
    # If the caller didn't pass speaker_wav we build per-speaker routing from
    # diarization data; if they did, we use it as a global override for all speakers.
    user_provided_speaker_wav = speaker_wav  # may be None

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    # --- MODIFIED: Added skipped flag for caching criteria ---
    # if wav_path.exists():
    #     return {
    #         "video_id": video_id,
    #         "audio_path": str(wav_path),
    #         "config": config,
    #         "skipped": True
    #     }

    source_path = str(trans_dir / f"{title}.json")

    # Load translated transcript to get per-segment speaker labels (from diarization).
    with open(source_path, "r", encoding="utf-8") as f:
        translated = json.load(f)
    segments = translated.get("segments", [])

    if user_provided_speaker_wav is not None:
        # Caller passed an explicit voice override — use it globally for all speakers.
        final_voice_routing = user_provided_speaker_wav
    else:
        # Build a per-speaker -> voice-WAV mapping so each diarized speaker gets
        # their own cloned voice (falls back to es/default.wav when no speaker-
        # specific WAV exists in pipeline_data/speakers/es/).
        unique_speakers = sorted(set(seg.get("speaker", "SPEAKER_00") for seg in segments))
        final_voice_routing = {
            spk: resolve_speaker_wav(settings.speakers_dir, "es", spk)
            for spk in unique_speakers
        }

    # Pass the mapping down the pipe (dict → per-speaker, str → global override)
    await _run_in_threadpool(
        None, svc.text_file_to_speech, source_path, str(audio_dir),
        alignment=alignment, speaker_wav=final_voice_routing,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
        "skipped": False,
    }

@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")
