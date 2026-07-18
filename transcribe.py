#!/usr/bin/env python3
"""Extract audio from a local video file and transcribe it to text.

Runs 100% locally using ffmpeg (audio extraction) and faster-whisper
(speech-to-text). No API keys, no tokens, no internet required after
the first run (the Whisper model is downloaded once and cached).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# Most common video container extensions
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".3gp", ".3g2", ".ts", ".mts",
    ".m2ts", ".vob", ".ogv", ".rm", ".rmvb", ".asf", ".f4v",
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]


def check_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit(
            "Error: ffmpeg is not installed or not on PATH.\n"
            "Install it with: brew install ffmpeg (macOS) "
            "or apt install ffmpeg (Linux)"
        )


def validate_video(path: Path) -> None:
    if not path.exists():
        sys.exit(f"Error: file not found: {path}")
    if not path.is_file():
        sys.exit(f"Error: not a file: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        sys.exit(
            f"Error: '{path.suffix}' is not a recognized video extension.\n"
            f"Supported: {supported}"
        )


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract audio as 16 kHz mono WAV (what Whisper expects)."""
    print(f"[1/3] Extracting audio from {video_path.name} ...")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",              # no video
            "-acodec", "pcm_s16le",
            "-ar", "16000",     # 16 kHz sample rate
            "-ac", "1",         # mono
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Error: ffmpeg failed to extract audio:\n{result.stderr[-2000:]}")


def format_timestamp(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def transcribe(audio_path: Path, model_name: str, language: str | None):
    from faster_whisper import WhisperModel

    print(f"[2/3] Loading Whisper model '{model_name}' (downloads once, then cached) ...")
    model = WhisperModel(model_name, device="auto", compute_type="auto")

    print("[3/3] Transcribing ...")
    segments, info = model.transcribe(str(audio_path), language=language)
    print(f"      Detected language: {info.language} "
          f"(confidence {info.language_probability:.0%})")

    results = []
    for seg in segments:
        results.append(seg)
        print(f"      [{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}] "
              f"{seg.text.strip()}")
    return results


def write_output(segments, output_path: Path, fmt: str) -> None:
    lines = []
    if fmt == "txt":
        lines = [seg.text.strip() for seg in segments]
    elif fmt == "srt":
        for i, seg in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(
                f"{format_timestamp(seg.start, srt=True)} --> "
                f"{format_timestamp(seg.end, srt=True)}"
            )
            lines.append(seg.text.strip())
            lines.append("")
    elif fmt == "vtt":
        lines.append("WEBVTT")
        lines.append("")
        for seg in segments:
            lines.append(
                f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}"
            )
            lines.append(seg.text.strip())
            lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract audio from a video and transcribe it locally (free, no API).",
    )
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=WHISPER_MODELS,
        help="Whisper model size (default: base). Bigger = more accurate but slower.",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Language code, e.g. 'en' or 'es' (default: auto-detect)",
    )
    parser.add_argument(
        "-f", "--format",
        default="txt",
        choices=["txt", "srt", "vtt"],
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path (default: <video name>.<format> next to the video)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Also save the extracted audio as a WAV next to the video",
    )
    args = parser.parse_args()

    check_ffmpeg()
    validate_video(args.video)

    output_path = args.output or args.video.with_suffix(f".{args.format}")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.wav"
        extract_audio(args.video, audio_path)

        if args.keep_audio:
            saved_audio = args.video.with_suffix(".wav")
            saved_audio.write_bytes(audio_path.read_bytes())
            print(f"      Saved extracted audio to {saved_audio}")

        segments = transcribe(audio_path, args.model, args.language)

    if not segments:
        print("Warning: no speech detected in the audio.")

    write_output(segments, output_path, args.format)
    print(f"\nDone! Transcript saved to: {output_path}")


if __name__ == "__main__":
    main()
