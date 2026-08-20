"""
Universal Data Compressor - Streamlit App (FIXED)
--------------------------------------------------
Upload video, audio, PDF, images, or any other file, choose a compression
strength and voice/audio settings, and get a compressed output.

WHAT WAS FIXED vs. the original version
----------------------------------------
1. compression_percentage now actually drives the OUTPUT SIZE, not just an
   encoder quality knob:
     - Video: two-pass (CPU) / CBR (NVENC/QSV) bitrate targeting so the
       output lands close to  original_size * (1 - compression_percentage/100).
     - Audio: bitrate is computed directly from the target size and clip
       duration instead of a fixed lookup table.
     - Images (JPEG/WEBP): binary search on quality until the encoded size
       is close to the target size.
     - Images (PNG): max lossless compression, then progressive downscale
       if still above target.
     - PDF: iterative embedded-image quality reduction until under target
       (or until it can't go lower — many PDFs are mostly text and simply
       can't shrink much, which is normal and expected).
2. UNIVERSAL SAFETY NET: after any compression path runs, the output size
   is compared to the original. If it's ever >= the original (this is what
   was causing "kabhi bada ho jata hai"), the app throws the compressed
   attempt away and ships the original file unchanged instead. The app
   will NEVER return a file larger than what you uploaded.
3. Honesty about generic files (zip, docx, mp3-inside-zip, exe, etc.):
   many file formats are ALREADY compressed internally. No algorithm can
   reliably shrink already-compressed/encrypted/random data by an arbitrary
   percentage without destroying it — that's a hard information-theory
   limit, not a bug. For those files this app tries gzip level 9 and, if
   that doesn't help, ships the original back with a note in the UI instead
   of corrupting or bloating the file.

Run with:  streamlit run app.py

IMPORTANT: The 1 TB upload limit is set in .streamlit/config.toml
(NOT in Python code — Streamlit does not support setting this at runtime
in a way that reliably takes effect, and st.file_uploader() has no
max_upload_size argument).
"""

import gzip
import io
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st

# Optional dependency for real PDF compression (falls back gracefully if absent)
try:
    import pikepdf

    PIKEPDF_AVAILABLE = True
except ImportError:
    PIKEPDF_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# --------------------------------------------------------------------------
# Helpers & Category Detection
# --------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTS = {".pdf"}


def format_bytes(num_bytes: float) -> str:
    """Human readable file size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:,.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:,.2f} PB"


def detect_category(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in PDF_EXTS:
        return "pdf"
    return "generic"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_duration(path: str) -> float:
    """Duration of a media file in seconds via ffprobe. Never returns 0."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15,
        )
        dur = float(result.stdout.strip())
        return dur if dur > 0.1 else 0.1
    except Exception:
        return 0.1


# --------------------------------------------------------------------------
# Voice / Audio parameter mapping (quality attributes only — bitrate for
# video/audio is now computed from the target size, see below)
# --------------------------------------------------------------------------

def voice_percentage_to_bitrate(voice_compression_percentage: int) -> str:
    """Used as a fallback bitrate and as the audio budget hint inside video."""
    max_kbps = 256
    min_kbps = 24
    pct = max(0, min(100, voice_compression_percentage))
    kbps = max_kbps - (pct / 100.0) * (max_kbps - min_kbps)
    kbps = int(round(kbps / 8.0) * 8)
    kbps = max(min_kbps, min(max_kbps, kbps))
    return f"{kbps}k"


def voice_percentage_to_sample_rate(voice_compression_percentage: int, voice_profile: str):
    if voice_profile == "Voice Only (Mono Speech, 16kHz)":
        return 16000
    if voice_profile == "Speech Standard (Mono Speech, 22kHz)":
        return 22050
    if voice_compression_percentage >= 70:
        return 16000
    if voice_compression_percentage >= 40:
        return 22050
    return None


def voice_percentage_to_channels(voice_compression_percentage: int, voice_profile: str):
    if voice_profile in (
            "Voice Only (Mono Speech, 16kHz)",
            "Speech Standard (Mono Speech, 22kHz)",
            "Compact Voice (Low Bitrate)",
    ):
        return 1
    if voice_compression_percentage >= 50:
        return 1
    return None


# --------------------------------------------------------------------------
# Hardware encoder detection
# --------------------------------------------------------------------------

def nvenc_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if "h264_nvenc" not in result.stdout:
            return False
    except Exception:
        return False
    try:
        gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        return gpu_check.returncode == 0
    except Exception:
        return False


def qsv_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if "h264_qsv" not in result.stdout:
            return False
    except Exception:
        return False
    try:
        null_out = "NUL" if os.name == "nt" else "/dev/null"
        test = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
             "-c:v", "h264_qsv", "-f", "null", null_out],
            capture_output=True, text=True, timeout=15,
        )
        return test.returncode == 0
    except Exception:
        return False


def get_video_encoder_mode() -> str:
    if nvenc_available():
        return "nvenc"
    if qsv_available():
        return "qsv"
    return "cpu"


# --------------------------------------------------------------------------
# Video encoding primitives (bitrate-targeted, so output size is predictable)
# --------------------------------------------------------------------------

# Codecs whose hardware decoder is broadly reliable on both NVENC and QSV.
# AV1 in particular was crashing QSV decode on some machines (and is one of
# the slowest formats to software-decode), so it's deliberately excluded —
# for AV1 we accept slower software decode rather than risk another crash.
_SAFE_HW_DECODE_CODECS = {"h264", "hevc", "h265", "mpeg2video", "vc1"}


def ffprobe_video_codec(path: str) -> str:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip().lower()
    except Exception:
        return ""


def _hwaccel_decode_args(encoder_mode: str, codec_name: str):
    """Only use hardware decode for codecs known to decode reliably on this
    encoder's hardware path. This is decided ONCE up front from the probed
    codec — not a retry — so a video is still only ever processed once."""
    if codec_name not in _SAFE_HW_DECODE_CODECS:
        return [], False
    if encoder_mode == "nvenc":
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"], True
    if encoder_mode == "qsv":
        return ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"], True
    return [], False


def _scale_filter(scale_factor, hw_active: bool, encoder_mode: str = "cpu"):
    if not scale_factor:
        return []
    expr = f"trunc(iw*{scale_factor}/2)*2:trunc(ih*{scale_factor}/2)*2"
    if hw_active and encoder_mode == "nvenc":
        return ["-vf", f"scale_cuda={expr}"]
    if hw_active and encoder_mode == "qsv":
        return ["-vf", f"scale_qsv={expr}"]
    return ["-vf", f"scale={expr}"]


def _audio_args(audio_bitrate_str, channels, sample_rate):
    args = ["-c:a", "aac", "-b:a", audio_bitrate_str]
    if channels is not None:
        args += ["-ac", str(channels)]
    if sample_rate is not None:
        args += ["-ar", str(sample_rate)]
    return args


def _run_video_cpu_single_pass(input_path, output_path, video_kbps, audio_bitrate_str,
                                channels, sample_rate, scale_factor):
    """CPU (libx264) single-pass VBV-capped encode — only encodes the file
    ONCE (not twice like two-pass). Uses the 'ultrafast' preset: since output
    size is already controlled by the bitrate cap (not by preset), spending
    extra time on a slower preset buys very little extra quality here, so
    ultrafast is the right trade for speed on large files."""
    cmd = ["ffmpeg", "-y", "-threads", "0", "-i", input_path,
           "-c:v", "libx264", "-b:v", f"{video_kbps}k",
           "-maxrate", f"{video_kbps}k", "-bufsize", f"{video_kbps * 2}k",
           "-preset", "ultrafast"]
    cmd += _scale_filter(scale_factor, hw_active=False)
    cmd += _audio_args(audio_bitrate_str, channels, sample_rate)
    cmd += [output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{r.stderr[-2000:]}")


def _run_video_cbr(input_path, output_path, encoder_mode, video_kbps, audio_bitrate_str,
                    channels, sample_rate, scale_factor):
    """NVENC / QSV single-pass CBR. Hardware DECODE is used only for codecs
    known to be reliable (h264/hevc/etc.) — decided once via a codec probe,
    not as a retry. Anything else (like AV1, which crashed QSV decode
    before) falls back to software decode for that one attempt, which is
    slower but reliable."""
    codec_name = ffprobe_video_codec(input_path)
    hwaccel_args, hw_active = _hwaccel_decode_args(encoder_mode, codec_name)

    if encoder_mode == "nvenc":
        codec_args = ["-c:v", "h264_nvenc"]
        preset_args = ["-preset", "p1"]
    else:  # qsv
        codec_args = ["-c:v", "h264_qsv"]
        preset_args = ["-preset", "veryfast", "-look_ahead", "0"]

    cmd = ["ffmpeg", "-y"] + hwaccel_args + ["-i", input_path]
    cmd += codec_args
    cmd += ["-rc", "cbr", "-b:v", f"{video_kbps}k",
            "-maxrate", f"{video_kbps}k", "-bufsize", f"{video_kbps * 2}k"]
    cmd += preset_args
    cmd += _scale_filter(scale_factor, hw_active=hw_active, encoder_mode=encoder_mode)
    cmd += _audio_args(audio_bitrate_str, channels, sample_rate)
    cmd += [output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg CBR encode ({encoder_mode}) failed:\n{r.stderr[-2000:]}")


def compress_video(input_path: str, output_path: str, compression_percentage: int,
                    downscale: bool, voice_profile: str, voice_compression_percentage: int):
    """
    Bitrate is derived from the requested TARGET SIZE
    (original_size * (1 - compression_percentage/100)), split between video
    and audio, then encoded in a SINGLE pass (VBV-capped on CPU, CBR on
    NVENC/QSV) — the video file is encoded exactly once, never twice.
    """
    original_size = os.path.getsize(input_path)
    duration = ffprobe_duration(input_path)

    sample_rate = voice_percentage_to_sample_rate(voice_compression_percentage, voice_profile)
    channels = voice_percentage_to_channels(voice_compression_percentage, voice_profile)

    scale_factor = None
    if downscale and compression_percentage >= 60:
        scale_factor = 0.85 - (compression_percentage - 60) / 100.0
        scale_factor = max(0.4, min(0.85, scale_factor))

    if compression_percentage <= 5:
        # Essentially "keep quality" — re-encode at high quality bitrate
        # instead of forcing a near-1:1 bitrate target (faster, looks better).
        video_kbps = max(500, int((original_size * 8 / 1000.0) / duration * 0.9))
        audio_bitrate_str = "192k"
    else:
        target_size = max(original_size * (1 - compression_percentage / 100.0), original_size * 0.02)
        target_total_kbits = target_size * 8 / 1000.0

        voice_kbps = int(voice_percentage_to_bitrate(voice_compression_percentage).rstrip("k"))
        audio_total_kbits = min(voice_kbps * duration, target_total_kbits * 0.25)
        audio_total_kbits = max(audio_total_kbits, 24 * duration)
        audio_kbps = max(24, int(audio_total_kbits / duration))
        audio_bitrate_str = f"{audio_kbps}k"

        video_total_kbits = max(target_total_kbits - audio_total_kbits, 40 * duration)
        video_kbps = max(40, int(video_total_kbits / duration))

    encoder_mode = get_video_encoder_mode()

    # Single attempt only — no retry/fallback re-encode. If hardware encoding
    # fails, the error surfaces directly instead of silently re-encoding the
    # video a second time.
    if encoder_mode == "cpu":
        _run_video_cpu_single_pass(input_path, output_path, video_kbps, audio_bitrate_str,
                                    channels, sample_rate, scale_factor)
    else:
        _run_video_cbr(input_path, output_path, encoder_mode, video_kbps, audio_bitrate_str,
                        channels, sample_rate, scale_factor)


def compress_audio(input_path: str, output_path: str, voice_profile: str,
                    voice_compression_percentage: int, compression_percentage: int):
    """
    Bitrate is computed directly from target size / duration, so
    "Compression Percentage" reliably controls the output size. The
    Voice/Audio profile and Voice Compression Percentage still control
    quality attributes (sample rate, mono/stereo).
    """
    original_size = os.path.getsize(input_path)
    duration = ffprobe_duration(input_path)
    sample_rate = voice_percentage_to_sample_rate(voice_compression_percentage, voice_profile)
    channels = voice_percentage_to_channels(voice_compression_percentage, voice_profile)

    if compression_percentage <= 5:
        audio_bitrate = "192k"
    else:
        target_size = max(original_size * (1 - compression_percentage / 100.0), original_size * 0.02)
        kbps = int((target_size * 8 / 1000.0) / duration)
        kbps = max(16, min(320, kbps))
        audio_bitrate = f"{kbps}k"

    cmd = ["ffmpeg", "-y", "-i", input_path, "-c:a", "aac", "-b:a", audio_bitrate]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    cmd += [output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr[-2000:]}")


def compress_pdf(input_path: str, output_path: str, compression_percentage: int):
    """
    Iteratively lowers embedded-image JPEG quality until the file is at or
    under the target size (or until we run out of quality steps — a mostly
    text PDF simply can't shrink much further, which is expected/normal).
    """
    if not PIKEPDF_AVAILABLE:
        raise RuntimeError("pikepdf is not installed. Run: pip install pikepdf")

    original_size = os.path.getsize(input_path)
    target_size = max(original_size * (1 - compression_percentage / 100.0), original_size * 0.05)

    if compression_percentage <= 5:
        quality_steps = [95]
    else:
        # Direct formula-based estimate first (usually enough on its own),
        # with at most ONE refinement pass instead of looping through many
        # full-document re-saves — this is what was making large PDFs slow.
        first_guess = max(10, int(90 - compression_percentage * 0.75))
        quality_steps = [first_guess, max(8, first_guess - 25)]

    for jpeg_quality in quality_steps:
        with pikepdf.open(input_path) as pdf:
            if PIL_AVAILABLE:
                for page in pdf.pages:
                    for _, image_obj in page.images.items():
                        try:
                            pdf_image = pikepdf.PdfImage(image_obj)
                            pil_img = pdf_image.as_pil_image()
                            if pil_img.mode in ("RGBA", "P"):
                                pil_img = pil_img.convert("RGB")
                            buf = io.BytesIO()
                            pil_img.save(buf, format="JPEG", quality=jpeg_quality)
                            buf.seek(0)
                            pikepdf.PdfImage.replace(pdf_image, buf.read())
                        except Exception:
                            continue
            pdf.save(output_path, compress_streams=True,
                     object_stream_mode=pikepdf.ObjectStreamMode.generate)
        if os.path.getsize(output_path) <= target_size:
            return
    # ran out of steps — output_path already holds the most-compressed attempt


def compress_image(input_path: str, output_path: str, compression_percentage: int):
    """
    JPEG/WEBP: binary search on quality until encoded size is close to
    target. PNG has no quality knob, so it's compressed losslessly and, if
    still above target, progressively downscaled.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is not installed. Run: pip install pillow")

    original_size = os.path.getsize(input_path)
    ext = Path(output_path).suffix.lower()

    img = Image.open(input_path)
    if img.mode in ("RGBA", "P") and ext in (".jpg", ".jpeg"):
        img = img.convert("RGB")

    if compression_percentage <= 5:
        if ext in (".jpg", ".jpeg", ".webp"):
            img.save(output_path, quality=95, optimize=True)
        else:
            img.save(output_path, optimize=True)
        return

    target_size = max(original_size * (1 - compression_percentage / 100.0), 2048)

    if ext in (".jpg", ".jpeg", ".webp"):
        fmt = "JPEG" if ext in (".jpg", ".jpeg") else "WEBP"
        lo, hi = 5, 95
        best_bytes = None
        tolerance = target_size * 0.07  # stop early once within ~7% of target
        for _ in range(6):
            mid = (lo + hi) // 2
            buf = io.BytesIO()
            img.save(buf, format=fmt, quality=mid)
            size = buf.tell()
            if best_bytes is None or abs(size - target_size) < abs(len(best_bytes) - target_size):
                best_bytes = buf.getvalue()
            if abs(size - target_size) <= tolerance:
                break
            if size > target_size:
                hi = mid - 1
            else:
                lo = mid + 1
            if lo > hi:
                break
        with open(output_path, "wb") as f:
            f.write(best_bytes)
    else:
        # PNG: lossless compress first, downscale progressively if needed.
        w, h = img.size
        scale = 1.0
        for _ in range(8):
            resized = img if scale == 1.0 else img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale)))
            )
            buf = io.BytesIO()
            resized.save(buf, format="PNG", optimize=True, compress_level=9)
            if buf.tell() <= target_size or scale <= 0.2:
                with open(output_path, "wb") as f:
                    f.write(buf.getvalue())
                return
            scale *= 0.85


def compress_generic_to_file(input_path: str, output_path: str, compression_percentage: int) -> None:
    """gzip is the best generic, format-agnostic option available. It will
    genuinely shrink text/uncompressed data but CANNOT shrink already
    compressed data (zip/docx/mp3/etc.) below its entropy floor — that's
    an information-theory limit, not a bug. Streams the file instead of
    loading it fully into RAM, and caps the gzip level at 6 (levels 7-9 cost
    a lot of extra time for very little extra size on large files)."""
    level = max(1, min(6, round(1 + (compression_percentage / 100) * 5)))
    with open(input_path, "rb") as f_in, gzip.open(output_path, "wb", compresslevel=level) as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)


# --------------------------------------------------------------------------
# Unified dispatcher
# --------------------------------------------------------------------------

def compress_file(input_path: str, tmpdir: str, original_filename: str,
                   compression_percentage: int, voice_compression_percentage: int,
                   voice_profile: str, downscale_video: bool):
    category = detect_category(original_filename)
    stem = Path(original_filename).stem
    ext = Path(original_filename).suffix
    original_size = os.path.getsize(input_path)

    if category == "video":
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg was not found on this system. Install ffmpeg to process videos.")
        out_name = f"{stem}_compressed{ext}"
        out_path = os.path.join(tmpdir, out_name)
        compress_video(
            input_path, out_path,
            compression_percentage=compression_percentage,
            downscale=downscale_video,
            voice_profile=voice_profile,
            voice_compression_percentage=voice_compression_percentage,
        )

    elif category == "audio":
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg was not found on this system. Install ffmpeg to process audio/voice.")
        out_name = f"{stem}_compressed.m4a"
        out_path = os.path.join(tmpdir, out_name)
        compress_audio(
            input_path, out_path,
            voice_profile=voice_profile,
            voice_compression_percentage=voice_compression_percentage,
            compression_percentage=compression_percentage,
        )

    elif category == "pdf":
        out_name = f"{stem}_compressed.pdf"
        out_path = os.path.join(tmpdir, out_name)
        compress_pdf(input_path, out_path, compression_percentage=compression_percentage)

    elif category == "image":
        out_name = f"{stem}_compressed{ext}"
        out_path = os.path.join(tmpdir, out_name)
        compress_image(input_path, out_path, compression_percentage=compression_percentage)

    else:
        gz_name = f"{stem}{ext}.gz"
        gz_path = os.path.join(tmpdir, gz_name)
        compress_generic_to_file(input_path, gz_path, compression_percentage)
        if os.path.getsize(gz_path) < original_size:
            out_name, out_path = gz_name, gz_path
        else:
            # Already-compressed / incompressible data: ship it back unchanged
            # rather than making it bigger. The "unchanged" file IS the
            # uploaded file itself (same tmpdir, same name) — point straight
            # at it instead of copying onto itself.
            out_name = f"{stem}{ext}"
            candidate_path = os.path.join(tmpdir, out_name)
            out_path = input_path if os.path.abspath(candidate_path) == os.path.abspath(input_path) else candidate_path
            if out_path != input_path:
                shutil.copyfile(input_path, out_path)
            try:
                os.remove(gz_path)
            except OSError:
                pass

    # ---- UNIVERSAL SAFETY NET ----
    # No matter what happened above, never return a file bigger than the
    # original. This is what stops the "kabhi bada ho jata hai" behaviour.
    if os.path.getsize(out_path) > original_size:
        fallback_name = f"{stem}{ext}"
        fallback_path = os.path.join(tmpdir, fallback_name)
        if os.path.abspath(fallback_path) == os.path.abspath(input_path):
            # The natural fallback name/location IS the uploaded file itself
            # (same tmpdir, same original filename) — nothing to copy, just
            # point straight at it and drop the oversized compressed attempt.
            if os.path.abspath(out_path) != os.path.abspath(input_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            out_path, out_name = input_path, fallback_name
        elif os.path.abspath(fallback_path) != os.path.abspath(out_path):
            shutil.copyfile(input_path, fallback_path)
            try:
                os.remove(out_path)
            except OSError:
                pass
            out_path, out_name = fallback_path, fallback_name

    return out_path, out_name, category


# --------------------------------------------------------------------------
# Streamlit UI Configuration & Custom HTML/CSS
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Zarp Offline- Data Compression Optimization Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
  :root {
    --navy-950: #0a1730;
    --navy-900: #0f2340;
    --navy-800: #152d54;
    --navy-700: #1d3c6c;
    --signal-blue: #1e5c97;
    --sky-cyan: #4fc3f7;
    --cloud-50: #f7f9fc;
    --line: #dde4ee;
    --slate-500: #5c6b81;
  }

  .stApp {
    background-color: var(--cloud-50) !important;
    font-family: 'Inter', sans-serif !important;
  }

  h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--navy-950) !important;
  }

  #MainMenu, footer,  { visibility: hidden; }

  .custom-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 40px;
    background: var(--navy-950);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin: -1rem -5rem 0rem -5rem;
    position: relative;
    z-index: 1;
  }
  .custom-header img { height: 42px; display: block; }

  .custom-hero {
    position: relative;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(30,92,151,0.55), transparent 60%), var(--navy-900);
    color: #fff;
    padding: 60px 20px 80px;
    text-align: center;
    margin: 0 -5rem 2rem -5rem;
  }
  .custom-hero h1 {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #ffffff !important;
  }
  .custom-hero h1 span {
    background: linear-gradient(100deg, #7fd8ff, #4fc3f7 45%, #8fb8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .custom-hero p {
    color: #a9bcd6;
    font-size: 16.5px;
    margin: 0 auto;
    max-width: 580px;
  }

  .stButton>button {
    width: 100%;
    background: #1e5c97 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 14px 28px !important;
    box-shadow: 0 10px 20px -10px rgba(30,92,151,0.5) !important;
  }
  .stButton>button:hover {
    background: #164a7a !important;
  }

  .stDownloadButton>button {
    width: 100%;
    background: linear-gradient(135deg, #2b8a3e, #1b5e20) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 14px 28px !important;
  }

   [data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid var(--line) !important;
  }

  /* -------- File uploader: normal dropzone, attractive Browse button -------- */
  [data-testid="stFileUploaderDropzoneInstructions"] div span {
    color: var(--slate-500) !important;
    font-weight: 400 !important;
    text-decoration: none !important;
  }
  [data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--slate-500) !important;
  }
  [data-testid="stFileUploaderDropzone"] button {
    background: #eaf4fc !important;
    color: var(--signal-blue) !important;
    border: 1px solid #bfe0f5 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: background 0.2s ease;
  }
  [data-testid="stFileUploaderDropzone"] button:hover {
    background: #d9ecfa !important;
  }


  .custom-footer {
    text-align: center;
    padding: 40px 20px 20px;
    color: var(--slate-500);
    font-size: 13.5px;
    border-top: 1px solid var(--line);
    margin-top: 60px;
  }

  /* -------- 3D feature / format cards -------- */
  .section-label {
    text-align: center;
    font-size: 13px;
    letter-spacing: 1.5px;
    font-weight: 700;
    color: var(--signal-blue);
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .section-title {
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    color: var(--navy-950);
    margin-bottom: 28px;
  }
  .feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 20px;
    margin: 0 0 40px 0;
    perspective: 900px;
  }
  .feature-card {
    background: linear-gradient(160deg, #ffffff 0%, #eef3fa 100%);
    border-radius: 18px;
    padding: 26px 14px 20px;
    text-align: center;
    border: 1px solid rgba(221,228,238,0.9);
    box-shadow:
      0 1px 0 rgba(255,255,255,0.9) inset,
      0 14px 24px -12px rgba(20,45,90,0.18),
      0 4px 8px -2px rgba(20,45,90,0.08);
    transition: transform 0.28s ease, box-shadow 0.28s ease;
    transform-style: preserve-3d;
  }
  .feature-card:hover {
    transform: translateY(-8px) rotateX(4deg) rotateY(-3deg) scale(1.03);
    box-shadow:
      0 1px 0 rgba(255,255,255,0.9) inset,
      0 26px 40px -14px rgba(20,45,90,0.28),
      0 8px 14px -4px rgba(30,92,151,0.18);
    border-color: var(--sky-cyan);
  }
  .feature-card .icon-badge {
    width: 54px;
    height: 54px;
    margin: 0 auto 14px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    background: linear-gradient(135deg, var(--signal-blue), var(--navy-800));
    box-shadow: 0 8px 16px -6px rgba(30,92,151,0.55);
  }
  .feature-card h4 {
    margin: 0 0 6px 0 !important;
    font-size: 14.5px !important;
    color: var(--navy-950) !important;
  }
  .feature-card p {
    margin: 0;
    font-size: 12px;
    color: var(--slate-500);
    line-height: 1.4;
  }

  /* -------- 3D report stat cards -------- */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 18px;
    margin: 6px 0 26px 0;
  }
  .stat-card {
    position: relative;
    background: linear-gradient(160deg, #ffffff 0%, #eef3fa 100%);
    border-radius: 18px;
    padding: 22px 22px 18px;
    border: 1px solid rgba(221,228,238,0.9);
    border-top: 4px solid var(--accent-color, var(--signal-blue));
    box-shadow:
      0 1px 0 rgba(255,255,255,0.9) inset,
      0 16px 28px -14px rgba(20,45,90,0.20),
      0 4px 10px -3px rgba(20,45,90,0.10);
    overflow: hidden;
  }
  .stat-card::before {
    content: "";
    position: absolute;
    top: -30%;
    right: -20%;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(79,195,247,0.25), transparent 70%);
  }
  .stat-card.accent-blue { --accent-color: #1e5c97; }
  .stat-card.accent-green { --accent-color: #2b8a3e; }
  .stat-card.accent-purple { --accent-color: #7c4dff; }
  .stat-card .stat-label {
    font-size: 12px;
    font-weight: 700;
    color: var(--slate-500);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }
  .stat-card .stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--navy-950);
  }
  .stat-card .stat-delta {
    display: inline-block;
    margin-top: 8px;
    font-size: 12.5px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
  }
  .stat-delta.positive { background: rgba(43,138,62,0.12); color: #1b5e20; }
  .stat-delta.neutral { background: rgba(30,92,151,0.10); color: var(--signal-blue); }
</style>

<div class="custom-header">
  <div><img src="https://assets.zyrosite.com/cdn-cgi/image/format=auto,w=375,fit=crop/YZ9byglzQnTGj8EP/zarp-labs-logo-jul-2024-AGB4Obj4LkSRGZ5J.png" alt="Zarp Labs"></div>
  <div><img src="https://assets.zyrosite.com/cdn-cgi/image/format=auto,w=768,h=401,fit=crop/YZ9byglzQnTGj8EP/asset-6new-XQNDAvZfAELCSVNr.png" alt="Zarp Offline"></div>
</div>

<div class="custom-hero">
  <h1>Zarp Offline <span>Data Compression Optimization Engine</span></h1>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Supported File Types — attractive 3D card grid
# --------------------------------------------------------------------------
st.markdown("""
<div class="section-label">What you can compress</div>
<div class="section-title">Supported File Types</div>
<div class="feature-grid">
  <div class="feature-card">
    <div class="icon-badge">🎬</div>
    <h4>Video</h4>
    <p>MP4, MOV, AVI, MKV, WEBM, FLV, WMV</p>
  </div>
  <div class="feature-card">
    <div class="icon-badge">🎙️</div>
    <h4>Audio &amp; Voice</h4>
    <p>MP3, WAV, M4A, AAC, OGG, FLAC, OPUS</p>
  </div>
  <div class="feature-card">
    <div class="icon-badge">🖼️</div>
    <h4>Images</h4>
    <p>JPG, PNG, BMP, TIFF, WEBP</p>
  </div>
  <div class="feature-card">
    <div class="icon-badge">📄</div>
    <h4>PDF</h4>
    <p>Shrinks embedded images &amp; streams</p>
  </div>
  <div class="feature-card">
    <div class="icon-badge">📝</div>
    <h4>Documents</h4>
    <p>DOC, DOCX, PPT, PPTX</p>
  </div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Main Application Execution & Sidebar
# --------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

st.sidebar.header("⚙️ General Settings")

compression_percentage = st.sidebar.slider(
    "Compression Percentage",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="Directly targets output size ≈ original_size × (1 - percentage/100) "
         "for video, audio, images, and PDFs (bitrate/quality auto-calculated "
         "to hit that target). For already-compressed generic files (zip, "
         "docx, mp3-in-zip, etc.) an exact percentage isn't always physically "
         "achievable — the app will never return a file bigger than the original.",
)

voice_compression_percentage = st.sidebar.slider(
    "Voice Compression Percentage",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="Controls audio quality attributes: sample rate and mono/stereo. "
         "Overall audio SIZE is driven by Compression Percentage above.",
)

st.sidebar.divider()

st.sidebar.header("🎙️ Voice & Audio Settings")

voice_profile = st.sidebar.selectbox(
    "Choose Voice / Audio Profile",
    options=[
        "Standard Full Stereo (Music & Video)",
        "Voice Only (Mono Speech, 16kHz)",
        "Speech Standard (Mono Speech, 22kHz)",
        "Compact Voice (Low Bitrate)",
        "High Quality Audio (192kbps Stereo)"
    ],
    index=0,
    help="Presets optimized for speech/voice recordings vs. standard stereo."
)
_preview_rate = voice_percentage_to_sample_rate(voice_compression_percentage, voice_profile)
_preview_channels = voice_percentage_to_channels(voice_compression_percentage, voice_profile)
st.sidebar.caption(
    "Effective audio quality: "
    + (f"{_preview_rate} Hz" if _preview_rate else "source sample rate")
    + (", mono" if _preview_channels == 1 else ", source channels")
    + " (bitrate is set automatically to hit your target size)"
)

st.sidebar.divider()
st.sidebar.subheader("Advanced Video Options")
downscale_video = st.sidebar.checkbox(
    "Downscale video resolution at high compression (≥60%)", value=True
)

# --------------------------------------------------------------------------
# File Uploader
# --------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a file",
    type=None
)
if uploaded_file is not None:
    category = detect_category(uploaded_file.name)
    st.write(f"**Detected type:** {category.capitalize()}  |  **Original size:** {format_bytes(uploaded_file.size)}")

    start = st.button("Compression", type="primary", use_container_width=True)

    if start:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, uploaded_file.name)
            with open(in_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            original_size = os.path.getsize(in_path)

            try:
                with st.spinner("Compressing file... Please wait."):
                    t0 = time.time()

                    out_path, out_name, category = compress_file(
                        in_path,
                        tmpdir,
                        uploaded_file.name,
                        compression_percentage=compression_percentage,
                        voice_compression_percentage=voice_compression_percentage,
                        voice_profile=voice_profile,
                        downscale_video=downscale_video,
                    )

                    elapsed = time.time() - t0

                compressed_size = os.path.getsize(out_path)
                with open(out_path, "rb") as f:
                    compressed_bytes = f.read()

                st.session_state.result = {
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                    "out_name": out_name,
                    "data": compressed_bytes,
                    "category": category,
                    "elapsed": elapsed,
                    "compression_percentage": compression_percentage,
                    "voice_compression_percentage": voice_compression_percentage,
                    "unchanged": compressed_size >= original_size,
                }
                st.success(f"Compression complete in {elapsed:.1f}s ✅")

            except Exception as e:
                st.session_state.result = None
                st.error(f"Compression failed: {e}")

# --------------------------------------------------------------------------
# Dashboard Reporting
# --------------------------------------------------------------------------

result = st.session_state.result
if result:
    st.divider()
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:4px 0 18px 0;">'
        '<span style="font-size:22px;">📊</span>'
        '<span style="font-family:\'Space Grotesk\',sans-serif;font-size:24px;'
        'font-weight:700;color:#0a1730;">Compression Report</span>'
        '</div>',
        unsafe_allow_html=True
    )

    original = result["original_size"]
    compressed = result["compressed_size"]
    reduction_pct = max(0.0, (1 - compressed / original) * 100) if original else 0.0
    remaining_pct = 100 - reduction_pct

    if result.get("unchanged"):
        st.warning(
            "This file's data was already at (or near) its compression limit "
            "(e.g. it's an already-compressed / encrypted / binary format), "
            "so it's returned unchanged rather than made larger."
        )

    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card accent-blue">
        <div class="stat-label">Original Size</div>
        <div class="stat-value">{format_bytes(original)}</div>
        <span class="stat-delta neutral">{result['category'].capitalize()}</span>
      </div>
      <div class="stat-card accent-green">
        <div class="stat-label">Compressed Size</div>
        <div class="stat-value">{format_bytes(compressed)}</div>
        <span class="stat-delta positive">-{reduction_pct:.1f}%</span>
      </div>
      <div class="stat-card accent-purple">
        <div class="stat-label">Space Saved</div>
        <div class="stat-value">{format_bytes(max(0, original - compressed))}</div>
        <span class="stat-delta positive">{result['elapsed']:.1f}s</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(1.0, remaining_pct / 100), text=f"Final file is {remaining_pct:.1f}% of original size")

    st.bar_chart(
        {"Size (bytes)": {"Original": original, "Compressed": compressed}},
    )

    st.write(
        f"Compression Percentage used: **{result['compression_percentage']}%** &nbsp;|&nbsp; "
        f"Voice Compression Percentage used: **{result['voice_compression_percentage']}%** &nbsp;|&nbsp; "
        f"Actual size reduction achieved: **{reduction_pct:.1f}%** &nbsp;|&nbsp; "
        f"Time taken: **{result['elapsed']:.1f}s**"
    )

    st.download_button(
        label="⬇️ Download Compressed File",
        data=result["data"],
        file_name=result["out_name"],
        use_container_width=True,
        type="primary",
    )

# --------------------------------------------------------------------------
# Footer HTML
# --------------------------------------------------------------------------

st.markdown("""
<div class="custom-footer">
  <div>Built with <span style="color:#e63946;">♥</span> by <b>Zarp Labs R&D</b></div>
  <div style="margin-top: 6px; color:#9aa7bb;">© Zarp Offline — Empowering the Last Mile</div>
</div>
""", unsafe_allow_html=True)