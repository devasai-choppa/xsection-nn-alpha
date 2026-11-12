"""Build the narrated demo from the Manim scenes and a pass over the dashboard.

``make video`` renders any missing 1080p60 scenes, records the Streamlit app
with Playwright, synthesizes the narration with edge-tts, and stitches the
timeline with MoviePy and ffmpeg. There is no music.

Outputs, all under ``demo/``:

- ``demo.mp4`` — 1920x1080, about two minutes, burned-in subtitles
- ``demo_short.mp4`` — 1080x1920, 45 seconds
- ``thumbnail.png``
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import edge_tts
from moviepy import AudioFileClip, VideoFileClip, concatenate_audioclips, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

from xsnn.config import load_config, project_root
from xsnn.log import setup_logging

logger = logging.getLogger("demo")

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
VOICE = "en-US-AndrewNeural"
FRAME = (1920, 1080)
FPS = 30

# (narration key, source). "card" is drawn. "dashboard" is the recording.
MAIN = [
    ("title", "card"),
    ("data", "DataToFeatures"),
    ("network", "ForwardPass"),
    ("walk", "WalkForward"),
    ("dashboard", "dashboard"),
    ("results", "RankingToPortfolio"),
    ("end", "card"),
]
SHORT = [
    ("short_title", "card"),
    ("short_network", "ForwardPass"),
    ("short_dashboard", "dashboard"),
    ("short_results", "RankingToPortfolio"),
    ("short_end", "card"),
]
SHORT_BUDGET = {
    "short_title": 5.0,
    "short_network": 12.0,
    "short_dashboard": 12.0,
    "short_results": 9.0,
    "short_end": 7.0,
}
CARDS: dict[str, list[str]] = {
    "title": [
        "xsection-nn-alpha",
        "Rank the cross-section.",
        "Long the top decile. Short the bottom.",
    ],
    "end": [
        "Not a trading system.",
        "Survivorship bias is in the universe.",
        "github.com/TechieGoku2623",
    ],
    "short_title": [
        "xsection-nn-alpha",
        "Long the top decile.",
        "Short the bottom.",
    ],
    "short_end": [
        "Research code, not a trading system.",
        "github.com/TechieGoku2623",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the narrated demo videos")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--voice", default=VOICE)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = project_root()
    setup_logging(Path(cfg["outputs"]["logs"]) / "video.log")
    work = root / "demo" / "build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    ensure_scenes(root)
    ensure_gifs(cfg)
    narration = parse_narration(root / "demo" / "narration.md")
    dashboard = record_dashboard(root, work / "dashboard.mp4", Path(cfg["outputs"]["figures"]), args.port)
    build_cut(
        root,
        work,
        narration,
        dashboard,
        MAIN,
        root / "demo" / "demo.mp4",
        voice=args.voice,
        exact=None,
    )
    build_cut(
        root,
        work / "short",
        narration,
        dashboard,
        SHORT,
        root / "demo" / "demo_short.mp4",
        voice=args.voice,
        exact=45.0,
        vertical=True,
    )
    write_thumbnail(scene_file(root, "ForwardPass"), root / "demo" / "thumbnail.png")
    verify_outputs(root)
    logger.info("Demo videos are in %s", root / "demo")


def ensure_scenes(root: Path) -> None:
    needed = {name for _key, name in MAIN + SHORT if name not in {"card", "dashboard"}}
    missing = sorted(name for name in needed if not scene_file(root, name).exists())
    if not missing:
        return
    logger.info("Rendering missing scenes at 1080p60: %s", ", ".join(missing))
    subprocess.run(
        [sys.executable, str(root / "scripts" / "render_animations.py"), "--quality", "h", "--skip-gifs"],
        cwd=root,
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "XSNN_ROOT": str(root)},
    )


def ensure_gifs(cfg: dict) -> None:
    folder = Path(cfg["outputs"]["reports"]) / "gifs"
    needed = ["rolling_rank_ic.gif", "decile_months.gif", "pnl_race.gif", "forward_pass.gif"]
    if all((folder / name).exists() for name in needed):
        return
    from xsnn.viz.animations import build_all_gifs

    build_all_gifs(cfg)


def scene_file(root: Path, name: str) -> Path:
    return root / "media" / "videos" / "manim_scenes" / "1080p60" / f"{name}.mp4"


def parse_narration(path: Path) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    parsed = {key: " ".join(part.strip() for part in lines if part.strip()) for key, lines in sections.items()}
    missing = [key for key, _source in MAIN + SHORT if key not in parsed or not parsed[key]]
    if missing:
        raise RuntimeError(f"Narration is missing sections: {missing}")
    return parsed


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def media_duration(path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        text=True,
    )
    return float(output.strip())


def run(command: list[str]) -> None:
    logger.debug(" ".join(command))
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        detail = (result.stderr or "")[-1500:]
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command[:6])}\n{detail}")


def synthesize(text: str, dest: Path, voice: str) -> None:
    """Synthesize one sentence, reusing a content-addressed cache across runs."""
    digest = hashlib.sha1(f"{voice}\n{text}".encode()).hexdigest()
    cache = project_root() / "demo" / ".tts-cache" / f"{digest}.mp3"
    if cache.exists() and cache.stat().st_size > 1000:
        shutil.copyfile(cache, dest)
        return

    async def _save(spoken: str, chosen: str) -> None:
        await edge_tts.Communicate(spoken, chosen).save(str(dest))

    voices = [voice, "en-US-GuyNeural", "en-US-JennyNeural"]
    last_error: Exception | None = None
    for attempt in range(6):
        chosen = voices[min(attempt // 2, len(voices) - 1)]
        dest.unlink(missing_ok=True)
        try:
            asyncio.run(_save(text, chosen))
            if dest.stat().st_size < 1000:
                raise RuntimeError("edge-tts wrote an empty file")
            cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dest, cache)
            time.sleep(0.35)
            return
        except Exception as exc:  # the speech service drops the occasional request
            last_error = exc
            logger.warning("edge-tts attempt %s failed for %r: %s", attempt + 1, text[:48], exc)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"edge-tts failed for {dest.name}") from last_error


def fit_video(src: Path, dest: Path, duration: float) -> None:
    """Scale to 1080p and trim or hold the last frame so the clip matches ``duration``."""
    pad = max(0.0, duration - media_duration(src))
    video_filter = (
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x101418,"
        f"fps={FPS}"
    )
    if pad > 0.04:
        video_filter += f",tpad=stop_mode=clone:stop_duration={pad:.3f}"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-t",
            f"{duration:.3f}",
            "-vf",
            video_filter,
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            str(dest),
        ]
    )


def fit_audio(src: Path, dest: Path, duration: float) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-t",
            f"{duration:.3f}",
            "-af",
            f"apad=whole_dur={duration:.3f}",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )


def render_card(lines: list[str], dest: Path, duration: float) -> None:
    image = Image.new("RGB", FRAME, (16, 20, 24))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(FONT_BOLD, 68)
    body_font = ImageFont.truetype(FONT, 36)
    heights = []
    fonts = []
    for index, line in enumerate(lines):
        font = title_font if index == 0 else body_font
        fonts.append(font)
        box = draw.textbbox((0, 0), line, font=font)
        heights.append(box[3] - box[1])
    gap = 28
    total = sum(heights) + gap * (len(lines) - 1)
    y = (FRAME[1] - total) / 2
    for line, font, height in zip(lines, fonts, heights, strict=True):
        box = draw.textbbox((0, 0), line, font=font)
        x = (FRAME[0] - (box[2] - box[0])) / 2
        color = (255, 255, 255) if font is title_font else (210, 214, 220)
        draw.text((x, y), line, font=font, fill=color)
        y += height + gap
    png = dest.with_suffix(".png")
    image.save(png)
    run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(png),
            "-t",
            f"{duration:.3f}",
            "-r",
            str(FPS),
            "-vf",
            "scale=1920:1080,format=yuv420p",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            str(dest),
        ]
    )


def source_for(root: Path, dashboard: Path, kind: str) -> Path:
    if kind == "dashboard":
        return dashboard
    if kind == "card":
        raise ValueError("cards are drawn, not loaded")
    path = scene_file(root, kind)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def visual_floor(root: Path, dashboard: Path, key: str, kind: str, exact: float | None) -> float:
    if exact is not None:
        return SHORT_BUDGET[key]
    if kind == "card":
        return 6.5
    if kind == "dashboard":
        return min(42.0, media_duration(dashboard))
    return media_duration(scene_file(root, kind))


def build_cut(
    root: Path,
    work: Path,
    narration: dict[str, str],
    dashboard: Path,
    sections: list[tuple[str, str]],
    dest: Path,
    *,
    voice: str,
    exact: float | None,
    vertical: bool = False,
) -> None:
    work.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    audios: list[Path] = []
    cues: list[tuple[float, float, str]] = []
    clock = 0.0
    for key, kind in sections:
        spoken = sentences(narration[key])
        sentence_files: list[Path] = []
        local = 0.0
        section_cues: list[tuple[float, float, str]] = []
        for index, sentence in enumerate(spoken):
            mp3 = work / f"{key}_{index}.mp3"
            synthesize(sentence, mp3, voice)
            length = media_duration(mp3)
            section_cues.append((local, local + length, sentence))
            local += length
            sentence_files.append(mp3)
        section_audio = work / f"{key}.wav"
        join_audio(sentence_files, section_audio)
        floor = visual_floor(root, dashboard, key, kind, exact)
        duration = floor if exact is not None else max(local, floor)
        clip = work / "clips" / f"{key}.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        if kind == "card":
            render_card(CARDS[key], clip, duration)
        else:
            fit_video(source_for(root, dashboard, kind), clip, duration)
        fitted_audio = work / f"{key}_fitted.wav"
        fit_audio(section_audio, fitted_audio, duration)
        for start, end, text in section_cues:
            if start >= duration:
                continue
            cues.append((clock + start, clock + min(end, duration), text))
        clips.append(clip)
        audios.append(fitted_audio)
        clock += duration
        logger.info("Section %s is %.1fs (speech %.1fs)", key, duration, local)
    silent = work / "silent.mp4"
    narration_wav = work / "narration.wav"
    concat_video(clips, silent)
    concat_audio(audios, narration_wav)
    if vertical:
        landscape = work / "landscape.mp4"
        mux(silent, narration_wav, landscape)
        framed = work / "vertical_nosub.mp4"
        to_vertical(landscape, framed)
        burn_vertical_subtitles(framed, cues, dest)
    else:
        subtitled = work / "subtitled.mp4"
        burn_subtitles(silent, cues, subtitled, vertical=False)
        mux(subtitled, narration_wav, dest)
    logger.info("Wrote %s (%.1fs, %.1f MB)", dest.name, media_duration(dest), dest.stat().st_size / 1e6)


def join_audio(files: list[Path], dest: Path) -> None:
    clips = [AudioFileClip(str(path)) for path in files]
    try:
        joined = concatenate_audioclips(clips)
        joined.write_audiofile(str(dest), fps=44100, logger=None)
    finally:
        for clip in clips:
            clip.close()


def concat_video(files: list[Path], dest: Path) -> None:
    clips = [VideoFileClip(str(path)) for path in files]
    try:
        joined = concatenate_videoclips(clips, method="chain")
        joined.write_videofile(
            str(dest),
            fps=FPS,
            codec="libx264",
            audio=False,
            preset="veryfast",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
            logger=None,
        )
    finally:
        for clip in clips:
            clip.close()


def concat_audio(files: list[Path], dest: Path) -> None:
    join_audio(files, dest)


def srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(cues: list[tuple[float, float, str]], dest: Path) -> None:
    blocks = []
    for index, (start, end, text) in enumerate(cues, start=1):
        blocks.append(f"{index}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}\n")
    dest.write_text("\n".join(blocks), encoding="utf-8")


def burn_subtitles(video: Path, cues: list[tuple[float, float, str]], dest: Path, *, vertical: bool) -> None:
    srt = dest.with_suffix(".srt")
    write_srt(cues, srt)
    # libass needs a path ffmpeg will not split on commas or colons.
    escaped = str(srt).replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")
    size = 15 if vertical else 22
    margin = 220 if vertical else 48
    style = (
        f"FontName=DejaVu Sans,FontSize={size},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,MarginV={margin}"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"subtitles='{escaped}':force_style='{style}'",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-an",
            str(dest),
        ]
    )


def mux(video: Path, audio: Path, dest: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            str(dest),
        ]
    )


def to_vertical(src: Path, dest: Path) -> None:
    video_filter = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:2[bg];"
        "[0:v]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto,fps=30"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-filter_complex",
            video_filter,
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(dest),
        ]
    )


def burn_vertical_subtitles(src: Path, cues: list[tuple[float, float, str]], dest: Path) -> None:
    """Burn captions onto the vertical frame so they sit in the lower third."""
    work = dest.parent / "build" / "vertical"
    work.mkdir(parents=True, exist_ok=True)
    silent = work / "silent.mp4"
    burned = work / "burned.mp4"
    run(["ffmpeg", "-y", "-i", str(src), "-an", "-c:v", "copy", str(silent)])
    burn_subtitles(silent, cues, burned, vertical=True)
    mux(burned, src, dest)


def write_thumbnail(scene: Path, dest: Path) -> None:
    frame = dest.with_name("_thumb_frame.png")
    run(["ffmpeg", "-y", "-ss", "3.5", "-i", str(scene), "-frames:v", "1", str(frame)])
    canvas = Image.new("RGB", (1280, 720), (16, 20, 24))
    shot = Image.open(frame).convert("RGB")
    shot.thumbnail((1180, 520))
    canvas.paste(shot, ((1280 - shot.width) // 2, 36))
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(FONT_BOLD, 42)
    caption = ImageFont.truetype(FONT, 24)
    draw.text((48, 600), "xsection-nn-alpha", font=title, fill=(255, 255, 255))
    draw.text((48, 656), "Walk-forward ranking of US large caps", font=caption, fill=(200, 204, 210))
    canvas.save(dest)
    frame.unlink(missing_ok=True)
    logger.info("Wrote thumbnail %s", dest)


def record_dashboard(root: Path, dest: Path, figures: Path, port: int) -> Path:
    """Drive the Streamlit app and return a silent 1080p recording."""
    log_path = dest.parent / "streamlit.log"
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["XSNN_ROOT"] = str(root)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(root / "app" / "streamlit_app.py"),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_server(port, process, log_path)
        webm = _playwright_pass(port, dest.parent, figures)
        fit_video(webm, dest, media_duration(webm))
        return dest
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log_handle.close()


def wait_for_server(port: int, process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.time() + 60
    url = f"http://127.0.0.1:{port}/_stcore/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Streamlit exited early.\n{log_path.read_text(encoding='utf-8')[-2000:]}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.4)
    raise RuntimeError("Streamlit did not become healthy")


def _playwright_pass(port: int, work: Path, figures: Path) -> Path:
    from playwright.sync_api import sync_playwright

    record_dir = work / "playwright"
    record_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(record_dir),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_selector("text=Ranked universe", timeout=90_000)
        page.evaluate(
            "() => { const h = document.querySelector('[data-testid=stHeader]'); if (h) h.style.display = 'none'; }"
        )
        page.wait_for_timeout(1200)
        figures.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(figures / "dashboard_rank.png"))
        _drive(page, figures)
        page.close()
        video_path = Path(page.video.path())
        context.close()
        browser.close()
    if not video_path.exists():
        raise RuntimeError("Playwright did not write a recording")
    return video_path


def _drive(page, figures: Path) -> None:
    panel = page.locator("[data-testid='stTabPanel']").first
    slider = panel.locator("input[type=range]")
    slider.focus()
    for _step in range(3):
        for _tick in range(6):
            slider.press("ArrowRight")
        page.wait_for_timeout(900)
    page.wait_for_timeout(800)
    _choose(page, panel.locator("[data-testid='stSelectbox'] input"), "AAPL")
    page.wait_for_timeout(1600)
    page.get_by_text("Interactive graph (pyvis)").click()
    page.wait_for_timeout(1600)
    page.locator("[data-testid='stMain']").evaluate("el => el.scrollTo(0, el.scrollHeight * 0.45)")
    page.wait_for_timeout(1400)
    page.screenshot(path=str(figures / "dashboard_attribution.png"))
    page.get_by_role("tab", name="Backtest").click()
    page.wait_for_timeout(1800)
    page.screenshot(path=str(figures / "dashboard_backtest.png"))
    backtest = page.locator("[data-testid='stTabPanel']").nth(1)
    cost = backtest.locator("input[type=range]")
    cost.focus()
    start_cost = cost.get_attribute("aria-valuetext") or cost.input_value()
    for _tick in range(20):
        cost.press("ArrowRight")
        page.wait_for_timeout(90)
    page.wait_for_timeout(2800)
    end_cost = cost.get_attribute("aria-valuetext") or cost.input_value()
    if start_cost == end_cost:
        raise RuntimeError(f"Cost slider did not move (still {start_cost})")
    page.screenshot(path=str(figures / "dashboard_backtest.png"))
    _choose(page, backtest.locator("[data-testid='stSelectbox'] input"), "LightGBM")
    page.wait_for_timeout(2200)
    page.get_by_role("tab", name="How it works").click()
    page.wait_for_timeout(3500)
    if page.locator("[data-testid='stException']").count():
        detail = page.locator("[data-testid='stException']").first.inner_text()
        raise RuntimeError(f"Streamlit raised during the demo pass:\n{detail[:1500]}")


def _choose(page, locator, value: str) -> None:
    locator.click()
    locator.fill(value)
    page.keyboard.press("Enter")
    page.wait_for_timeout(400)


def verify_outputs(root: Path) -> None:
    demo = root / "demo" / "demo.mp4"
    short = root / "demo" / "demo_short.mp4"
    thumb = root / "demo" / "thumbnail.png"
    for path in (demo, short, thumb):
        if not path.exists() or path.stat().st_size < 10_000:
            raise RuntimeError(f"Missing or empty output: {path}")
    _expect_video(demo, 1920, 1080, min_seconds=80, max_bytes=100 * 1024 * 1024)
    _expect_video(short, 1080, 1920, min_seconds=43.5, max_seconds=47.0)
    with Image.open(thumb) as image:
        if image.size[0] < 640 or image.size[1] < 360:
            raise RuntimeError(f"Thumbnail is too small: {image.size}")
    mean = _mean_volume(demo)
    if mean < -45:
        raise RuntimeError(f"Demo audio looks silent (mean volume {mean:.1f} dB)")
    logger.info("Verified demo.mp4, demo_short.mp4, and thumbnail.png")


def _expect_video(
    path: Path,
    width: int,
    height: int,
    *,
    min_seconds: float,
    max_seconds: float | None = None,
    max_bytes: int | None = None,
) -> None:
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    payload = json.loads(probe)
    stream = payload["streams"][0]
    duration = float(payload["format"]["duration"])
    if int(stream["width"]) != width or int(stream["height"]) != height:
        raise RuntimeError(f"{path.name} is {stream['width']}x{stream['height']}, expected {width}x{height}")
    if duration < min_seconds or (max_seconds is not None and duration > max_seconds):
        raise RuntimeError(f"{path.name} duration {duration:.2f}s is outside the expected range")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise RuntimeError(f"{path.name} is {path.stat().st_size / 1e6:.1f} MB")
    audio = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if "aac" not in audio.stdout:
        raise RuntimeError(f"{path.name} has no AAC audio")


def _mean_volume(path: Path) -> float:
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", result.stderr)
    if match is None:
        raise RuntimeError("volumedetect did not report a mean volume")
    return float(match.group(1))


if __name__ == "__main__":
    main()
