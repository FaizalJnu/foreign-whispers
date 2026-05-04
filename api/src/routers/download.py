"""POST /api/download — download YouTube video + captions (issue by5)."""

import json
import pathlib

from fastapi import APIRouter, HTTPException, Request

from api.src.core.config import settings
from api.src.core.video_registry import get_video
from api.src.schemas.download import CaptionSegment, DownloadRequest, DownloadResponse
from api.src.services.download_service import DownloadService

router = APIRouter(prefix="/api")

_download_service = DownloadService(ui_dir=settings.data_dir)


def _video_id_from_url(url: str) -> str | None:
    """Extract the 11-char YouTube video ID from a URL without a network call."""
    import re
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None


@router.post("/download", response_model=DownloadResponse)
async def download_endpoint(body: DownloadRequest):
    """Download video and captions, returning video_id and caption segments.

    For videos already in the registry the video_id and title are resolved
    locally (no yt_dlp network call).  yt_dlp is only called when the video
    is not registered or when files are actually missing.
    """
    # --- Fast path: resolve from registry without a network call ---
    video_id = _video_id_from_url(body.url)
    entry = get_video(video_id) if video_id else None

    if entry:
        stem = entry.title
        videos_dir = settings.videos_dir
        captions_dir = settings.youtube_captions_dir
        videos_dir.mkdir(parents=True, exist_ok=True)
        captions_dir.mkdir(parents=True, exist_ok=True)

        video_path = videos_dir / f"{stem}.mp4"
        caption_path = captions_dir / f"{stem}.txt"

        # Only call yt_dlp if we actually need to download something
        if not video_path.exists():
            _download_service.download_video(body.url, str(videos_dir), stem)

        if not caption_path.exists():
            _download_service.download_caption(body.url, str(captions_dir), stem)

        segments = _download_service.read_caption_segments(caption_path)
        return DownloadResponse(
            video_id=video_id,
            title=entry.title,
            caption_segments=segments,
        )

    # --- Slow path: unknown video — must call yt_dlp to get metadata ---
    video_id, title = _download_service.get_video_info(body.url)

    # Re-check registry now that we have the ID
    entry = get_video(video_id)
    stem = entry.title if entry else title.replace(":", "")

    videos_dir = settings.videos_dir
    captions_dir = settings.youtube_captions_dir
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    video_path = videos_dir / f"{stem}.mp4"
    caption_path = captions_dir / f"{stem}.txt"

    if not video_path.exists():
        _download_service.download_video(body.url, str(videos_dir), stem)

    if not caption_path.exists():
        _download_service.download_caption(body.url, str(captions_dir), stem)

    segments = _download_service.read_caption_segments(caption_path)

    return DownloadResponse(
        video_id=video_id,
        title=title,
        caption_segments=segments,
    )

