# VideoAudioExtractor

Extract audio from a local video and transcribe it to text — **100% local and free**.
No API keys, no tokens, no cloud. Uses [ffmpeg](https://ffmpeg.org/) for audio
extraction and [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(OpenAI's Whisper model running on your own machine) for transcription.

## Requirements

- Python 3.9–3.13 (3.14 not yet supported by faster-whisper)
- ffmpeg on your PATH (`brew install ffmpeg` on macOS)

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Basic — writes my_video.txt next to the video
python transcribe.py my_video.mp4

# Choose model size (bigger = more accurate, slower)
python transcribe.py my_video.mkv --model small

# Force a language instead of auto-detect
python transcribe.py my_video.avi --language es

# Subtitle output (SRT or VTT with timestamps)
python transcribe.py my_video.mov --format srt

# Custom output path, and keep the extracted WAV audio
python transcribe.py my_video.webm -o transcript.txt --keep-audio
```

## Supported video formats

mp4, mkv, avi, mov, wmv, flv, webm, m4v, mpg, mpeg, 3gp, 3g2, ts, mts,
m2ts, vob, ogv, rm, rmvb, asf, f4v

## Model sizes

| Model | Speed | Accuracy | Download size |
|-------|-------|----------|---------------|
| tiny | fastest | lowest | ~75 MB |
| base (default) | fast | good | ~145 MB |
| small | medium | better | ~485 MB |
| medium | slow | great | ~1.5 GB |
| large-v3 / turbo | slowest | best | ~3 GB / ~1.6 GB |

The model is downloaded once on first use and cached in
`~/.cache/huggingface/` — after that everything runs fully offline.
