"""AI review of the best candidates: a local vision-language model watches each one and judges it.

Fully offline. llama.cpp's `llama-server` (Vulkan build, MIT) runs as its own
process -- none of its DLLs share our process -- serving Qwen3-VL-4B-Instruct
(Apache-2.0, Q4_K_M, 2.5 GB + 0.45 GB vision projector) on 127.0.0.1.

For every candidate, strongest first, it gets a DOSSIER: frames spread over the
moment (starting a little before it), the player's words, the game's dialogue
(subtitles and transcript), the notifications read on screen, the sounds and
what the picture looked like. It answers in JSON: a 0-10 score, the event, a
title and a one-line description for the short, and where the action really
starts. The number reviewed scales with the recording's length; `skip` (a
threading.Event) ends the review early and keeps what is done.
"""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .. import video_io
from . import models

RUNTIME = {
    "name": "llama.cpp b12645 (Vulkan)",
    "url": "https://github.com/ggml-org/llama.cpp/releases/download/b12645/llama-b12645-bin-win-vulkan-x64.zip",
    "sha256": "5284ca8ed5064c3b401f936d1348116e215c334cf89e473e23b52e906a5cf6bd",
    "size_mb": 39,
    "licence": "MIT",
}
MODELS = {
    "4b": {
        "name": "Qwen3-VL-4B-Instruct",
        "repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
        "revision": "1cd86afb9a95c410a6038ab3b40d8b578c892266",
        "files": {
            "Qwen3VL-4B-Instruct-Q4_K_M.gguf": "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a",
            "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf": "30ba2c7dd3127a4561b6cba9d13d0f711c91bdb38742e2f56d73c8cb596bd06d",
        },
        "size_mb": 2950,
        "licence": "Apache-2.0",
    },
    "8b": {
        "name": "Qwen3-VL-8B-Instruct",
        "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "revision": "f982a07559d4a2f6c8744d840bf6fccab30eea96",
        "files": {
            "Qwen3VL-8B-Instruct-Q4_K_M.gguf": "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2",
            "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf": "c6ba85508d82f42590e6eb77d5340369ab6fecf107a7561d809523d8aa5f3bfd",
        },
        "size_mb": 5780,
        "licence": "Apache-2.0",
    },
}
MODEL = MODELS["4b"]
PROMPT_VERSION = "v3"   # part of the cache key: a new prompt is a new review


def model_for(cfg: dict) -> dict:
    return MODELS.get(cfg.get("model", "4b"), MODEL)


def model_id(model: dict) -> str:
    return model["name"].lower().replace("-", "") + "-q4km-" + PROMPT_VERSION
EVENTS = ("chase", "escape", "crash", "shootout", "explosion", "stunt", "funny", "fail", "death", "arrest",
          "mission", "exploration", "dialogue", "other")
SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "event": {"type": "string", "enum": list(EVENTS)},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "action_starts_at": {"type": "integer"},
        "best_moment_at": {"type": "integer"},
        "why": {"type": "string"},
    },
    "required": ["score", "event", "title", "description", "action_starts_at", "best_moment_at", "why"],
}
PAIR_SCHEMA = {
    "type": "object",
    "properties": {"better": {"type": "string", "enum": ["A", "B"]}, "why": {"type": "string"}},
    "required": ["better", "why"],
}


def _folder() -> Path:
    return models.models_dir() / "judge"


def ensure(log, model: dict = MODEL) -> tuple[Path, Path, Path]:
    """(llama-server.exe, model, projector), downloading once with SHA-256 checks."""
    folder = _folder()
    server = folder / "llama" / "llama-server.exe"
    if not server.is_file():
        log(f"   ⬇️  Downloading the AI review runtime ({RUNTIME['size_mb']} MB, once)…")
        part = models._download(RUNTIME["url"], RUNTIME["sha256"], folder / "llama.part")
        with zipfile.ZipFile(part) as archive:
            archive.extractall(folder / "llama")
        part.unlink()
    paths = []
    for name, sha in model["files"].items():
        path = folder / name
        if not path.is_file():
            log(f"   ⬇️  Downloading the AI review model {name} (once)…")
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"
            models._download(url, sha, folder / f"{name}.part").replace(path)
        paths.append(path)
    return server, paths[0], paths[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    """llama-server on a free local port, for the duration of one review."""

    def __init__(self, exe: Path, model: Path, mmproj: Path, cfg: dict) -> None:
        self.port = _free_port()
        self.args = [str(exe), "-m", str(model), "--mmproj", str(mmproj), "-ngl", "99", "-c", str(cfg["context"]),
                     "--host", "127.0.0.1", "--port", str(self.port), "-np", "1"]
        self.process: subprocess.Popen | None = None
        self.log_file = model.parent / "llama-server.log"

    def start(self, cancel: video_io.Canceller, timeout: float = 180) -> None:
        log = self.log_file.open("w", encoding="utf-8", errors="replace")
        self.process = subprocess.Popen(self.args, stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=video_io._NO_WINDOW, cwd=str(Path(self.args[0]).parent))
        deadline = time.time() + timeout
        while time.time() < deadline:
            cancel.check()
            if self.process.poll() is not None:
                raise RuntimeError(f"the AI review runtime stopped while loading (see {self.log_file})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as reply:
                    if reply.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        raise RuntimeError("the AI review model did not load in time")

    def ask(self, content: list[dict], max_tokens: int, timeout: float = 180, schema: dict | None = None,
            raw: bool = False) -> dict:
        body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": max_tokens,
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "review", "schema": schema or SCHEMA}}}
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            data = json.loads(reply.read())
        text = data["choices"][0]["message"]["content"]
        return json.loads(text[text.find("{"):text.rfind("}") + 1]) if raw else parse(text)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(10)


def parse(text: str) -> dict:
    """The model's JSON, forgiving stray text around it."""
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    data["score"] = max(0, min(10, int(data.get("score", 0))))
    if data.get("event") not in EVENTS:
        data["event"] = "other"
    for key in ("title", "description", "why"):
        data[key] = str(data.get(key, "")).strip()
    return data


def frames(path: Path, start: float, end: float, count: int, width: int, cancel: video_io.Canceller) -> list[bytes]:
    """`count` JPEG frames spread evenly over [start, end), in one ffmpeg call."""
    length = max(0.5, end - start)
    args = [video_io.ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, start):.2f}", "-i", str(path),
            "-t", f"{length:.2f}", "-map", "0:v:0", "-vf", f"fps={count / length:.4f},scale={width}:-2",
            "-frames:v", str(count), "-q:v", "5", "-f", "image2pipe", "-c:v", "mjpeg", "-"]
    code, out, _err = video_io._capture(args, cancel, text=False)
    if code != 0:
        return []
    pictures, cursor = [], 0
    while True:
        begin = out.find(b"\xff\xd8", cursor)
        finish = out.find(b"\xff\xd9", begin + 2)
        if begin < 0 or finish < 0:
            break
        pictures.append(out[begin:finish + 2])
        cursor = finish + 2
    return pictures


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def dossier(moment, start: float, end: float, heard, screen, game: str) -> str:
    """Everything known about [start, end), as text the model can read."""
    lines = [f"Game: {game}. This is a candidate clip from a gameplay recording, from {_clock(start)} to "
             f"{_clock(end)} ({end - start:.0f} s). Second 0 below is {_clock(start)}."]
    player = [w for w in heard.player_words if start <= w.start < end]
    story = [w for w in heard.game_words if start <= w.start < end]
    if player:
        lines.append("The player says: " + " ".join(w.text for w in player)[:600])
    elif heard.mode == "none":
        lines.append("The player is not talking in this recording.")
    if story:
        lines.append("Game characters say (audio): " + " ".join(w.text for w in story)[:600])
    if screen is not None:
        subtitles = [l["text"] for l in screen.near(start, end) if l["zone"] == "subtitle"]
        notes = [l["text"] for l in screen.near(start, end) if l["zone"] != "subtitle" and len(l["text"]) > 12]
        if subtitles:
            lines.append("Subtitles on screen: " + " | ".join(dict.fromkeys(subtitles))[:600])
        if notes:
            lines.append("Notifications on screen: " + " | ".join(dict.fromkeys(notes))[:400])
    if moment.explanation:
        lines.append("Automatic detectors heard/saw: " + moment.explanation)
    if getattr(moment, "marked", None) is not None:
        lines.append('The player said "clipar agora" (clip now) right after this clip: they found what just '
                     "happened worth clipping. Judge those seconds and say where that action began.")
    return "\n".join(lines)


PROMPT = """{dossier}

The {count} images are frames from this clip in order, at seconds: {times}.

You are choosing clips for YouTube Shorts / TikTok from a {game} gameplay recording. Judge only what the frames and
the text above show; do not invent events. Score 0-10 how good this clip is as a short on its own:
 0-2 nothing happens (driving around, walking, menus, talking with nothing happening)
 3-4 mildly interesting
 5-6 something happens (a chase, a fight, a stunt, a funny line) but it is ordinary
 7-8 exciting or funny: a viewer would watch to the end
 9-10 spectacular: a crash, a big escape, an explosion, a wild stunt, a great reaction
Give the event type; a short catchy title in natural Brazilian Portuguese, the way a Brazilian creator would post it
(max 60 characters, at most one emoji, no hashtags, no literal translations, nothing the frames do not show);
a one-sentence description in Brazilian Portuguese; the second (0-{last}) where the action actually begins; the
second (0-{last}) of the single most striking moment (it will open the short as a teaser); and a short reason in
English. Good titles look like: "Fugi da polícia e capotei no final 😱", "Pulei da ponte e me arrependi",
"Essa perseguição acabou do pior jeito", "O caminhão voou longe!", "Ninguém esperava esse final" -- write your
own for what these frames show; never reuse an example."""


def review(path: Path, duration: float, moments: list, heard, screen, cfg: dict, cache: Path, *, log, progress,
           cancel: video_io.Canceller, skip: threading.Event | None = None) -> dict:
    """Review the strongest candidates in order; returns a summary. Moments get .ai and a new .score."""
    budget = int(min(cfg["max_reviews"], max(cfg["min_reviews"], round(duration / 60 / cfg["minutes_per_review"]))))
    marked = [m for m in moments if getattr(m, "marked", None) is not None]
    others = sorted((m for m in moments if getattr(m, "marked", None) is None), key=lambda m: -m.score)
    queue_ = marked + others[:max(0, budget - len(marked))]   # what the player marked is always reviewed
    model = model_for(cfg)
    store_file = cache / f"judge-{model_id(model)}.json"
    store = json.loads(store_file.read_text(encoding="utf-8")) if store_file.is_file() else {}
    todo = [m for m in queue_ if _key(m) not in store]
    summary = {"model": model["name"], "planned": len(queue_), "reviewed": 0, "skipped": False, "seconds": 0.0}
    server = None
    started = time.perf_counter()
    try:
        if todo:
            exe, weights, mmproj = ensure(log, model)
            log(f"   🤖 AI review of the best {len(queue_)} moments ({model['name']}, offline)…")
            server = Server(exe, weights, mmproj, cfg)
            server.start(cancel)
        for index, moment in enumerate(queue_, start=1):
            cancel.check()
            if skip is not None and skip.is_set():
                summary["skipped"] = True
                log("   ⏭️  AI review stopped early: using what is reviewed so far")
                break
            key = _key(moment)
            if key not in store:
                tick = time.perf_counter()
                try:
                    store[key] = _judge(server, path, moment, heard, screen, cfg, cancel)
                except video_io.Cancelled:
                    raise
                except Exception as error:  # noqa: BLE001 -- one bad answer does not stop the review
                    log(f"   ⚠️  AI review of {_clock(moment.start)} failed: {error}")
                    continue
                store_file.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
                each = time.perf_counter() - tick
                left = sum(1 for m in queue_[index:] if _key(m) not in store) * each
                log(f"   🤖 {index}/{len(queue_)} · {store[key]['score']}/10 {store[key]['title']} · "
                    f"~{max(0, left) / 60:.0f} min left")
            _apply(moment, store[key], cfg)
            summary["reviewed"] += 1
            progress(index, len(queue_))
        if server is not None and not summary["skipped"] and cfg.get("rerank_top", 0) > 1:
            summary["reranked"] = _rerank(server, path, moments, store, store_file, cfg, log, cancel)
    finally:
        if server is not None:
            server.stop()
    summary["seconds"] = round(time.perf_counter() - started, 1)
    return summary


def _key(moment) -> str:
    return f"{moment.start:.1f}-{moment.end:.1f}"


def _judge(server: Server, path: Path, moment, heard, screen, cfg: dict, cancel) -> dict:
    lead = float(cfg["look_before_s"])
    start, end = max(0.0, moment.start - lead), moment.end
    pictures = frames(path, start, end, int(cfg["frames"]), int(cfg["frame_width"]), cancel)
    if not pictures:
        raise RuntimeError("no frames")
    length = end - start
    times = [round(i * length / len(pictures)) for i in range(len(pictures))]
    text = PROMPT.format(dossier=dossier(moment, start, end, heard, screen, cfg["game"]), count=len(pictures),
                         times=", ".join(map(str, times)), game=cfg["game"], last=int(length))
    content = [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p).decode()}}
        for p in pictures]
    answer = server.ask(content, int(cfg["max_tokens"]))
    answer["window_start"] = start
    return answer


def _apply(moment, answer: dict, cfg: dict) -> None:
    """Blend the AI score into the ranking and, when it makes sense, move the start to where the action begins."""
    moment.ai = {k: answer.get(k) for k in ("score", "event", "title", "description", "why")}
    if "best_moment_at" in answer:
        best = answer.get("window_start", moment.start) + float(answer["best_moment_at"])
        # The teaser shows something from the clip itself, never from the seconds before it.
        moment.ai["best_at"] = round(min(max(best, moment.start + 0.75), moment.end - 0.75), 2)
    weight = float(cfg["weight"])
    base = moment.detector_score or moment.score
    moment.score = int(round(weight * answer["score"] * 10 + (1 - weight) * base))
    begins = answer.get("window_start", moment.start) + float(answer.get("action_starts_at", 0))
    proposed = max(0.0, begins - float(cfg["pre_roll_s"]))
    if moment.start - float(cfg["look_before_s"]) <= proposed < moment.end - 8 and abs(proposed - moment.start) >= 2:
        moment.ai["start_moved_from"] = moment.start
        moment.start = round(proposed, 2)


PAIR_PROMPT = """Two candidate clips from the same {game} recording. The first {a} images are clip A in order, the next
{b} are clip B. Which one makes the better YouTube Short / TikTok on its own -- more exciting, funnier or more
spectacular from the first seconds? Answer A or B and why, briefly. Clip A: {ta}. Clip B: {tb}."""


def _rerank(server: Server, path: Path, moments: list, store: dict, store_file: Path, cfg: dict, log, cancel) -> int:
    """One bubble pass over the top reviewed moments, comparing neighbours two at a time.

    Absolute 0-10 scores drift from one answer to the next; "which of these two
    is better" is steadier. Each swap trades the two moments' scores, so the
    order everywhere (cards, curve) follows the comparison.
    """
    top = sorted((m for m in moments if m.ai), key=lambda m: -m.score)[: int(cfg["rerank_top"])]
    if len(top) < 2:
        return 0
    log(f"   🤖 Comparing the top {len(top)} two by two…")
    swaps = 0
    for i in range(len(top) - 1):
        cancel.check()
        a, b = top[i], top[i + 1]
        key = f"pair:{_key(a)}|{_key(b)}"
        if key not in store:
            pics = [frames(path, m.start, m.end, 4, 384, cancel) for m in (a, b)]
            if not pics[0] or not pics[1]:
                continue
            text = PAIR_PROMPT.format(game=cfg["game"], a=len(pics[0]), b=len(pics[1]),
                                      ta=a.ai.get("title", ""), tb=b.ai.get("title", ""))
            content = [{"type": "text", "text": text}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p).decode()}}
                for p in pics[0] + pics[1]]
            try:
                store[key] = server.ask(content, 150, schema=PAIR_SCHEMA, raw=True)
            except Exception as error:  # noqa: BLE001
                log(f"   ⚠️  Comparison skipped: {error}")
                continue
            store_file.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
        if store[key].get("better") == "B":
            a.score, b.score = b.score, a.score
            top[i], top[i + 1] = b, a
            swaps += 1
    return swaps
