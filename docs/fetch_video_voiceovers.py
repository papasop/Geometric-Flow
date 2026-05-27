#!/usr/bin/env python3
"""Generate prebuilt Mandarin voiceover tracks for IsitHUB videos.

This is the "方案 C" batch path: GitHub Actions reads video items, tries to
fetch public YouTube captions, translates them to Chinese, generates MP3 files,
and publishes a static manifest. The browser never sees API keys.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NEWS_PATH = ROOT / "docs" / "news.json"
MANIFEST_PATH = ROOT / "docs" / "video-voiceover.json"
OUT_DIR = ROOT / "docs" / "audio" / "voiceovers"

MAX_VIDEOS = int(os.getenv("VOICEOVER_MAX_VIDEOS", "6"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("VOICEOVER_MAX_TRANSCRIPT_CHARS", "18000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "coral")
VOICEOVER_AUDIO_FALLBACK = os.getenv("VOICEOVER_AUDIO_FALLBACK", "1") == "1"
VOICEOVER_MAX_AUDIO_SECONDS = int(os.getenv("VOICEOVER_MAX_AUDIO_SECONDS", "600"))
VOICEOVER_FORCE = os.getenv("VOICEOVER_FORCE", "0") == "1"
YOUTUBE_COOKIES_FILE = os.getenv("YOUTUBE_COOKIES_FILE", "").strip()
YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES", "").strip()

YOUTUBE_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})")
VTT_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}")
TAG_RE = re.compile(r"<[^>]+>")
LIVE_RE = re.compile(r"\b(live|直播|watch live|business news live|market news live|newshour)\b", re.I)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def video_id_from_url(url: str) -> str:
    match = YOUTUBE_RE.search(url or "")
    return match.group(1) if match else ""


def collect_videos() -> list[dict[str, Any]]:
    payload = load_json(NEWS_PATH, {})
    sections = payload.get("sections") or {}
    candidates: list[dict[str, Any]] = []
    for section_id in ("video", "hot"):
        candidates.extend((sections.get(section_id) or {}).get("items") or [])

    def is_live_item(item: dict[str, Any]) -> bool:
        haystack = " ".join(str(item.get(key) or "") for key in ("title", "titleEn", "titleZh", "source", "summary"))
        return bool(LIVE_RE.search(haystack))

    candidates = sorted(candidates, key=lambda item: 1 if is_live_item(item) else 0)
    seen: set[str] = set()
    videos: list[dict[str, Any]] = []
    for item in candidates:
        url = str(item.get("url") or item.get("sourceUrl") or "")
        video_id = video_id_from_url(url)
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        live = is_live_item(item)
        if live:
            continue
        videos.append({
            "id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": item.get("title") or item.get("titleEn") or item.get("titleZh") or video_id,
            "titleZh": item.get("titleZh") or "",
            "source": item.get("source") or "YouTube",
            "isLive": live,
        })
        if len(videos) >= MAX_VIDEOS:
            break
    return videos


def clean_vtt(text: str) -> str:
    lines: list[str] = []
    recent: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "WEBVTT" or line.startswith(("Kind:", "Language:", "NOTE")):
            continue
        if line.isdigit() or VTT_TIME_RE.match(line):
            continue
        line = TAG_RE.sub("", line)
        line = line.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line in recent:
            continue
        lines.append(line)
        recent.append(line)
        if len(recent) > 5:
            recent.pop(0)
    return " ".join(lines)


def youtube_cookie_args(tmp: str | Path) -> list[str]:
    if YOUTUBE_COOKIES_FILE:
        cookie_path = Path(YOUTUBE_COOKIES_FILE)
        if cookie_path.exists():
            return ["--cookies", str(cookie_path)]
    if YOUTUBE_COOKIES:
        cookie_path = Path(tmp) / "youtube-cookies.txt"
        cookie_path.write_text(YOUTUBE_COOKIES + "\n", encoding="utf-8")
        return ["--cookies", str(cookie_path)]
    return []


def fetch_public_caption(video: dict[str, Any]) -> tuple[str, str]:
    if not shutil.which("yt-dlp"):
        return "", "yt-dlp is not installed"
    with tempfile.TemporaryDirectory() as tmp:
        command = [
            "yt-dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs", "en.*,en,zh-Hans,zh-Hant,zh.*",
            "--sub-format", "vtt",
            "--no-warnings",
            "-o", str(Path(tmp) / "%(id)s.%(ext)s"),
            video["url"],
        ]
        command[1:1] = youtube_cookie_args(tmp)
        result = subprocess.run(command, text=True, capture_output=True, timeout=150)
        files = sorted(Path(tmp).glob("*.vtt"))
        if not files:
            return "", (result.stderr or result.stdout or "no public captions found").strip()[-240:]
        preferred = next((path for path in files if ".en" in path.name), files[0])
        return clean_vtt(preferred.read_text(encoding="utf-8", errors="ignore")), ""


def split_text(text: str, limit: int = 2800) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    while text:
        part = text[:limit]
        cut = max(part.rfind("."), part.rfind("?"), part.rfind("!"), part.rfind("。"), part.rfind("？"), part.rfind("！"))
        if cut > limit * 0.45:
            part = part[:cut + 1]
        chunks.append(part.strip())
        text = text[len(part):].strip()
    return chunks


def openai_post(endpoint: str, payload: dict[str, Any]) -> bytes:
    req = urllib.request.Request(
        f"https://api.openai.com/v1/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def openai_multipart(endpoint: str, fields: dict[str, str], files: dict[str, Path]) -> bytes:
    boundary = f"----IsitHUB{int(time.time() * 1000)}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for name, path in files.items():
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
            .encode("utf-8")
        )
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    req = urllib.request.Request(
        f"https://api.openai.com/v1/{endpoint}",
        data=b"".join(chunks),
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return response.read()


def download_audio_sample(video: dict[str, Any], target: Path) -> tuple[bool, str]:
    if not shutil.which("yt-dlp"):
        return False, "yt-dlp is not installed"
    command = [
        "yt-dlp",
        "--no-warnings",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "64K",
        "--download-sections",
        f"*00:00:00-00:{VOICEOVER_MAX_AUDIO_SECONDS // 60:02d}:{VOICEOVER_MAX_AUDIO_SECONDS % 60:02d}",
        "-o",
        str(target.with_suffix(".%(ext)s")),
        video["url"],
    ]
    command[1:1] = youtube_cookie_args(target.parent)
    result = subprocess.run(command, text=True, capture_output=True, timeout=300)
    mp3 = target.with_suffix(".mp3")
    if mp3.exists() and mp3.stat().st_size > 1024:
        mp3.replace(target)
        return True, ""
    return False, (result.stderr or result.stdout or "audio download failed").strip()[-240:]


def transcribe_audio(path: Path) -> str:
    raw = openai_multipart("audio/transcriptions", {
        "model": OPENAI_TRANSCRIBE_MODEL,
        "response_format": "json",
    }, {"file": path})
    payload = json.loads(raw.decode("utf-8"))
    return str(payload.get("text") or "").strip()


def translate_to_chinese(text: str) -> str:
    raw = openai_post("responses", {
        "model": OPENAI_TEXT_MODEL,
        "input": (
            "Translate this English video transcript into natural spoken Mandarin Chinese. "
            "Preserve names, companies, tickers, and numbers. Do not add commentary.\n\n"
            f"{text}"
        ),
    })
    payload = json.loads(raw.decode("utf-8"))
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    parts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def text_to_speech(text: str, path: Path) -> None:
    audio = openai_post("audio/speech", {
        "model": OPENAI_TTS_MODEL,
        "voice": OPENAI_TTS_VOICE,
        "input": text,
        "response_format": "mp3",
    })
    path.write_bytes(audio)


def concat_mp3(parts: list[Path], output: Path) -> None:
    if len(parts) == 1:
        shutil.copyfile(parts[0], output)
        return
    if not shutil.which("ffmpeg"):
        shutil.copyfile(parts[0], output)
        return
    with tempfile.TemporaryDirectory() as tmp:
        list_path = Path(tmp) / "files.txt"
        list_path.write_text("".join(f"file '{part.as_posix()}'\n" for part in parts), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output)], check=True, capture_output=True)


def generate(video: dict[str, Any], prior: dict[str, Any] | None) -> dict[str, Any]:
    video_id = video["id"]
    audio_rel = f"audio/voiceovers/{video_id}.zh.mp3"
    audio_path = ROOT / "docs" / audio_rel
    base = {**video, "audio": audio_rel, "updatedAt": now_iso()}
    if audio_path.exists() and not VOICEOVER_FORCE:
        return {**(prior or base), **base, "status": "ready"}
    if not OPENAI_API_KEY:
        return {**base, "status": "missing_openai_key", "note": "Set OPENAI_API_KEY to generate Chinese voiceover audio."}

    transcript, reason = fetch_public_caption(video)
    transcript = transcript[:MAX_TRANSCRIPT_CHARS].strip()
    if len(transcript) < 120:
        if not VOICEOVER_AUDIO_FALLBACK:
            return {**base, "status": "no_caption", "note": reason or "No public caption transcript found."}
        with tempfile.TemporaryDirectory() as tmp:
            audio_sample = Path(tmp) / f"{video_id}.source.mp3"
            ok, audio_reason = download_audio_sample(video, audio_sample)
            if not ok:
                return {**base, "status": "no_audio", "note": audio_reason or reason or "No caption or downloadable audio found."}
            transcript = transcribe_audio(audio_sample)[:MAX_TRANSCRIPT_CHARS].strip()
        if len(transcript) < 120:
            return {**base, "status": "transcript_empty", "note": "Audio transcription returned too little text."}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts: list[Path] = []
        for index, chunk in enumerate(split_text(transcript), start=1):
            zh = translate_to_chinese(chunk)
            if not zh:
                continue
            part_path = Path(tmp) / f"{video_id}.{index}.mp3"
            text_to_speech(zh, part_path)
            parts.append(part_path)
            time.sleep(0.2)
        if not parts:
            return {**base, "status": "translation_empty", "note": "OpenAI returned no translated text."}
        concat_mp3(parts, audio_path)
    return {**base, "status": "ready", "segments": len(parts), "transcriptChars": len(transcript)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prior = load_json(MANIFEST_PATH, {"items": []})
    prior_by_id = {str(item.get("id")): item for item in prior.get("items", []) if item.get("id")}
    items: list[dict[str, Any]] = []
    for video in collect_videos():
        try:
            items.append(generate(video, prior_by_id.get(video["id"])))
        except Exception as error:  # Keep the batch moving; surface status in JSON.
            items.append({**video, "audio": f"audio/voiceovers/{video['id']}.zh.mp3", "status": "error", "note": str(error)[-240:], "updatedAt": now_iso()})

    payload = {
        "meta": {
            "updatedAt": now_iso(),
            "source": "YouTube public captions + OpenAI translation/TTS",
            "translationModel": OPENAI_TEXT_MODEL,
            "transcriptionModel": OPENAI_TRANSCRIBE_MODEL,
            "ttsModel": OPENAI_TTS_MODEL,
            "audioFallback": VOICEOVER_AUDIO_FALLBACK,
            "youtubeCookies": bool(YOUTUBE_COOKIES or YOUTUBE_COOKIES_FILE),
            "maxVideos": MAX_VIDEOS,
        },
        "items": items,
    }
    write_json(MANIFEST_PATH, payload)
    ready = sum(1 for item in items if item.get("status") == "ready")
    print(f"wrote {MANIFEST_PATH} with {ready}/{len(items)} ready voiceovers")


if __name__ == "__main__":
    main()
