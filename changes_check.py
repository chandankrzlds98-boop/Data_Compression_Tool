"""
Universal Data Compressor - Streamlit App
------------------------------------------
Upload video, audio, PDF, images, or any other file, choose a compression
strength and voice/audio settings, and get a compressed output.

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


# --------------------------------------------------------------------------
# Voice / Audio parameter mapping
# --------------------------------------------------------------------------
# These functions turn "Voice Compression Percentage" (10-90) into concrete
# ffmpeg parameters, so the slider actually changes the output instead of
# just changing a label in the UI.

def voice_percentage_to_bitrate(voice_compression_percentage: int) -> str:
    """
    Higher percentage -> smaller/more-compressed audio -> lower bitrate.
    Maps 10-90 (%) onto a 256k-24k bitrate range.
    """
    max_kbps = 256
    min_kbps = 24
    pct = max(0, min(100, voice_compression_percentage))
    kbps = max_kbps - (pct / 100.0) * (max_kbps - min_kbps)
    kbps = int(round(kbps / 8.0) * 8)  # round to a "nice" multiple of 8
    kbps = max(min_kbps, min(max_kbps, kbps))
    return f"{kbps}k"


def voice_percentage_to_sample_rate(voice_compression_percentage: int, voice_profile: str):
    """
    Higher percentage -> more aggressive downsampling.
    Explicit voice profiles still take priority when they specify a rate.
    """
    if voice_profile == "Voice Only (Mono Speech, 16kHz)":
        return 16000
    if voice_profile == "Speech Standard (Mono Speech, 22kHz)":
        return 22050

    if voice_compression_percentage >= 70:
        return 16000
    if voice_compression_percentage >= 40:
        return 22050
    return None  # keep source sample rate


def voice_percentage_to_channels(voice_compression_percentage: int, voice_profile: str):
    """
    Higher percentage -> collapse to mono to save space.
    """
    if voice_profile in (
            "Voice Only (Mono Speech, 16kHz)",
            "Speech Standard (Mono Speech, 22kHz)",
            "Compact Voice (Low Bitrate)",
    ):
        return 1
    if voice_compression_percentage >= 50:
        return 1
    return None  # keep source channel count


# --------------------------------------------------------------------------
# Compression Strategies
# --------------------------------------------------------------------------

def nvenc_available() -> bool:
    """Checks both that ffmpeg supports nvenc AND that a working Nvidia GPU/driver exists."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        if "h264_nvenc" not in result.stdout:
            return False
    except Exception:
        return False

    # Confirm an actual driver/GPU is present, not just compiled-in support
    try:
        gpu_check = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        return gpu_check.returncode == 0
    except Exception:
        return False


def qsv_available() -> bool:
    """
    Checks whether ffmpeg supports Intel Quick Sync Video (h264_qsv) AND
    that it can actually be initialized on this machine (Intel iGPU + driver
    present). This is a real encode test, not just a string match, because
    h264_qsv can appear in -encoders even when the QSV device can't init.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        if "h264_qsv" not in result.stdout:
            return False
    except Exception:
        return False

    try:
        # Encode a tiny throwaway clip to /dev/null (or NUL on Windows) to
        # confirm the QSV device actually initializes on this machine.
        null_out = "NUL" if os.name == "nt" else "/dev/null"
        test = subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                "-c:v", "h264_qsv", "-f", "null", null_out,
            ],
            capture_output=True, text=True, timeout=15
        )
        return test.returncode == 0
    except Exception:
        return False


def get_video_encoder_mode() -> str:
    """Priority: nvenc > qsv > cpu. Checked in order, first available wins."""
    if nvenc_available():
        return "nvenc"
    if qsv_available():
        return "qsv"
    return "cpu"


def compress_video(input_path: str, output_path: str, compression_percentage: int,
                    downscale: bool, voice_profile: str, voice_compression_percentage: int):
    """
    Uses NVENC (Nvidia GPU) when available, falls back to Intel Quick Sync
    (QSV) when available, and falls back to libx264 (CPU) otherwise.
    Also retries on CPU automatically if the hardware path fails at runtime.
    Higher compression_percentage -> higher CQ/QP (hardware) or CRF (x264) -> smaller file.
    """
    cq = int(18 + (compression_percentage / 100) * 22)
    audio_bitrate = voice_percentage_to_bitrate(voice_compression_percentage)
    sample_rate = voice_percentage_to_sample_rate(voice_compression_percentage, voice_profile)
    channels = voice_percentage_to_channels(voice_compression_percentage, voice_profile)

    encoder_mode = get_video_encoder_mode()  # "nvenc", "qsv", or "cpu"

    scale_factor = None
    if downscale and compression_percentage >= 60:
        scale_factor = 0.85 - (compression_percentage - 60) / 100.0  # ~0.85 down to ~0.5
        scale_factor = max(0.4, min(0.85, scale_factor))

    cmd = ["ffmpeg", "-y"]

    if encoder_mode == "nvenc":
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif encoder_mode == "qsv":
        cmd += ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]

    cmd += ["-i", input_path]

    if encoder_mode == "nvenc":
        cmd += [
            "-c:v", "h264_nvenc",
            "-preset", "p1",   # p1 = fastest NVENC preset (p7 = slowest/best quality)
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
        ]
    elif encoder_mode == "qsv":
        cmd += [
            "-c:v", "h264_qsv",
            "-preset", "veryfast",  # QSV presets: veryfast..veryslow
            "-global_quality", str(cq),
            "-look_ahead", "0",     # off = faster, less encode-time analysis
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-crf", str(cq),
            "-preset", "veryfast",
            "-threads", "0",
        ]

    cmd += ["-c:a", "aac", "-b:a", audio_bitrate]

    if channels is not None:
        cmd += ["-ac", str(channels)]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]

    if scale_factor is not None:
        if encoder_mode == "nvenc":
            cmd += ["-vf", f"scale_cuda=trunc(iw*{scale_factor}/2)*2:trunc(ih*{scale_factor}/2)*2"]
        elif encoder_mode == "qsv":
            cmd += ["-vf", f"scale_qsv=trunc(iw*{scale_factor}/2)*2:trunc(ih*{scale_factor}/2)*2"]
        else:
            cmd += ["-vf", f"scale=trunc(iw*{scale_factor}/2)*2:trunc(ih*{scale_factor}/2)*2"]

    cmd += [output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if encoder_mode in ("nvenc", "qsv"):
            # Hardware path failed at runtime — retry once with CPU encoding
            cpu_cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-c:v", "libx264", "-crf", str(cq),
                "-preset", "ultrafast", "-threads", "0",
                "-c:a", "aac", "-b:a", audio_bitrate,
            ]
            if channels is not None:
                cpu_cmd += ["-ac", str(channels)]
            if sample_rate is not None:
                cpu_cmd += ["-ar", str(sample_rate)]
            if scale_factor is not None:
                cpu_cmd += ["-vf", f"scale=trunc(iw*{scale_factor}/2)*2:trunc(ih*{scale_factor}/2)*2"]
            cpu_cmd += [output_path]

            retry = subprocess.run(cpu_cmd, capture_output=True, text=True)
            if retry.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed (both {encoder_mode.upper()} and CPU fallback):\n{retry.stderr[-2000:]}"
                )
        else:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")


def compress_audio(input_path: str, output_path: str, voice_profile: str, voice_compression_percentage: int):
    """
    Recompress dedicated audio/voice files using ffmpeg AAC encoder.
    Bitrate, sample rate, and channel count are all driven dynamically by
    voice_compression_percentage (and refined by the chosen voice_profile).
    """
    audio_bitrate = voice_percentage_to_bitrate(voice_compression_percentage)
    sample_rate = voice_percentage_to_sample_rate(voice_compression_percentage, voice_profile)
    channels = voice_percentage_to_channels(voice_compression_percentage, voice_profile)

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:a", "aac", "-b:a", audio_bitrate,
    ]

    if channels is not None:
        cmd += ["-ac", str(channels)]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]

    cmd += [output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")


def compress_pdf(input_path: str, output_path: str, compression_percentage: int):
    if not PIKEPDF_AVAILABLE:
        raise RuntimeError("pikepdf is not installed. Run: pip install pikepdf")

    jpeg_quality = max(10, 95 - int(compression_percentage * 0.8))

    with pikepdf.open(input_path) as pdf:
        if PIL_AVAILABLE and compression_percentage >= 30:
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

        pdf.save(
            output_path,
            compress_streams=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )


def compress_image(input_path: str, output_path: str, compression_percentage: int):
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is not installed. Run: pip install pillow")

    quality = max(5, 95 - int(compression_percentage * 0.85))
    img = Image.open(input_path)
    if img.mode in ("RGBA", "P") and Path(output_path).suffix.lower() in (".jpg", ".jpeg"):
        img = img.convert("RGB")

    save_kwargs = {}
    ext = Path(output_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        save_kwargs = {"quality": quality, "optimize": True}
    elif ext == ".png":
        save_kwargs = {"optimize": True, "compress_level": min(9, 1 + compression_percentage // 11)}
    elif ext == ".webp":
        save_kwargs = {"quality": quality}

    img.save(output_path, **save_kwargs)


def compress_generic(input_path: str, output_path: str, compression_percentage: int):
    level = max(1, min(9, round(1 + (compression_percentage / 100) * 8)))
    with open(input_path, "rb") as f_in, gzip.open(output_path, "wb", compresslevel=level) as f_out:
        shutil.copyfileobj(f_in, f_out)


# --------------------------------------------------------------------------
# Unified dispatcher — this is the single entry point the UI calls.
# It ensures compression_percentage and voice_compression_percentage are
# always threaded through to the real algorithm for the detected file type.
# --------------------------------------------------------------------------

def compress_file(input_path: str, tmpdir: str, original_filename: str,
                  compression_percentage: int, voice_compression_percentage: int,
                  voice_profile: str, downscale_video: bool):
    category = detect_category(original_filename)
    stem = Path(original_filename).stem
    ext = Path(original_filename).suffix

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
        # DOC/DOCX, PPT/PPTX, CSV/XLSX, ZIP, and any other file type
        out_name = f"{stem}{ext}.gz"
        out_path = os.path.join(tmpdir, out_name)
        compress_generic(input_path, out_path, compression_percentage=compression_percentage)

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

# Sidebar Controls
st.sidebar.header("⚙️ General Settings")

compression_percentage = st.sidebar.slider(
    "Compression Percentage",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="Higher = smaller file, lower overall quality. Drives video CRF/"
         "resolution, image quality, PDF image quality, and archive level.",
)

voice_compression_percentage = st.sidebar.slider(
    "Voice Compression Percentage",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="Higher = smaller/more compressed audio. Drives audio bitrate, "
         "sample rate, and mono/stereo channel count for audio and video files.",
)

st.sidebar.divider()

# Sidebar Audio & Voice Settings
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
    help="Presets optimized for speech/voice recordings vs. standard stereo. "
         "Combined with Voice Compression Percentage above."
)
# Live preview of the derived audio parameters so users can see the effect
# of Voice Compression Percentage before running compression.
_preview_bitrate = voice_percentage_to_bitrate(voice_compression_percentage)
_preview_rate = voice_percentage_to_sample_rate(voice_compression_percentage, voice_profile)
_preview_channels = voice_percentage_to_channels(voice_compression_percentage, voice_profile)
st.sidebar.caption(
    f"Effective audio target: {_preview_bitrate} bitrate"
    + (f", {_preview_rate} Hz" if _preview_rate else ", source sample rate")
    + (", mono" if _preview_channels == 1 else ", source channels")
)

st.sidebar.divider()
st.sidebar.subheader("Advanced Video Options")
downscale_video = st.sidebar.checkbox(
    "Downscale video resolution at high compression (≥60%)", value=True
)

# --------------------------------------------------------------------------
# File Uploader
# --------------------------------------------------------------------------
# NOTE: The actual upload size ceiling (1 TB) is configured in
# .streamlit/config.toml via [server] maxUploadSize = 1048576.
# It is intentionally not surfaced anywhere in this UI.

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
                # noinspection PyPackageRequirements
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