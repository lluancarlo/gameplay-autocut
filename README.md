<div align="center">
  <img src="gameplay-autocut-logo.png" alt="Gameplay AutoCut" width="180">

  # Gameplay AutoCut

  ### A totally offline solution to transform gameplay into viral clips

  Long gameplay recording in, ready-to-post vertical shorts out. No cloud, no uploads, no subscription.
</div>

---

Gameplay AutoCut finds the best moments of a gameplay recording and turns the ones
you pick into **1080×1920 vertical shorts** for YouTube Shorts, TikTok and Reels.
It runs on Windows, entirely on your own PC, and is tuned for **GTA 6** for now.

![Picking the best moments of a recording](docs/pick-clips.png)

## How it works

The window walks you through four steps:

1. **Video**: choose or drop a recording.
2. **Find clips**: Gameplay AutoCut listens to the audio, looks at the picture 10 times a
   second and reads text on screen. A local AI then watches the best candidates
   and scores each one 0–10, with a title and a description.
3. **Pick clips**: preview each moment, trim it, and tick the ones you want.
4. **Make shorts**: check the vertical framing and generate the clips. A
   `_titles.txt` file is written next to them, ready to paste when you post.

Your voice is detected automatically, whether it is on its own audio track or
mixed with the game. Menus, maps and pause screens are left out.

## Run from source

Python 3.12, NVIDIA or other DirectX 12 GPU recommended.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip uninstall -y onnxruntime
pip install -r requirements-vision.txt
pip install --no-deps -r requirements-ocr.txt
pip install -r requirements-cuda.txt     # optional: Whisper on an NVIDIA GPU
python GameplayAutoCut.py
```

The first analysis downloads the models (~5 GB) into `GameplayAutoCut-data\`. A
recording that was analysed before opens almost instantly.

## Build the .exe

```
pyinstaller GameplayAutoCut.spec          # -> publish\GameplayAutoCut.exe
publish\GameplayAutoCut.exe --self-test   # checks every model; writes GameplayAutoCut-data\self-test.json
```

The app is portable. Settings go in `GameplayAutoCut.ini` and models and cache
in `GameplayAutoCut-data\`, both next to the `.exe`. Delete them and nothing is left
behind.

## What's inside

| Part | Used for |
|---|---|
| ffmpeg (imageio-ffmpeg) | decoding, cutting, NVENC encoding |
| Whisper (faster-whisper) | speech and its language |
| CED (sherpa-onnx) | sounds: gunfire, engines, sirens, crashes, shouts |
| SigLIP 2 (ONNX Runtime, DirectML) | what the picture shows |
| RapidOCR | text on screen ("A polícia está procurando você", subtitles…) |
| Qwen3-VL-4B (llama.cpp) | the AI review of the best moments |
| PySide6 | the window |

Tuning lives in `config/moments.json` and `config/shorts.json`. The design notes
are in [`docs/ai-highlights/`](docs/ai-highlights/).

## Command line

```
python find-moments.py recording.mp4          # table of the best moments
python make-short.py clip.mp4                 # one clip -> vertical short
python -m pytest -q                           # tests
```

## License

MIT. See [LICENSE](LICENSE).
