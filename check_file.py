"""
Universal Data Compressor - Streamlit App
------------------------------------------
Upload video, PDF, images, or any other file, choose a compression
method, and get a compressed output plus a dashboard showing how
much space was saved.

Run with:  streamlit run app.py
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
# Helpers
# --------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
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
    if ext in IMAGE_EXTS:
        return "image"
    if ext in PDF_EXTS:
        return "pdf"
    return "generic"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ---- Advanced Video Compression Engine -----------------------------------

def compress_video_advanced(
    input_path: str,
    output_path: str,
    codec: str,
    method: str,
    pct_val: int,
    mb_val: float,
    crf_val: int,
    preset_speed: str,
    resolution: str,
    max_bitrate: int,
    downscale: bool
):
    """FFmpeg wrapper accommodating codecs, target methods, speeds, and resolutions."""

    # Map Codecs
    vcodec = "libx264"
    if "H.265" in codec or "HEVC" in codec:
        vcodec = "libx265"
    elif "AV1" in codec:
        vcodec = "libsvtav1"

    cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", vcodec]

    # Map Preset Speed
    speed_map = {
        "Very fast (Default)": "veryfast",
        "Fast": "fast",
        "Medium": "medium",
        "Slow - best compression": "slow"
    }
    cmd += ["-preset", speed_map.get(preset_speed, "medium")]

    # Method Logic
    if method == "Target a file size (Percentage)":
        crf = int(18 + (pct_val / 100) * 22)
        cmd += ["-crf", str(crf)]
        if downscale and pct_val >= 60:
            cmd += ["-vf", "scale=trunc(iw*0.7/2)*2:trunc(ih*0.7/2)*2"]

    elif method == "Target a file size (MB)":
        try:
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprintwrappers=1:nokey=1", input_path]
            duration = float(subprocess.check_output(probe_cmd).decode().strip())
            target_bits = mb_val * 8 * 1024 * 1024
            bitrate_kbps = int((target_bits / duration) / 1000)
            cmd += ["-b:v", f"{bitrate_kbps}k"]
        except Exception:
            cmd += ["-crf", "26"]

    elif method == "Target a video quality":
        cmd += ["-crf", str(crf_val)]

    elif method == "Target a video resolution":
        res_map = {
            "1080p (1920×1080)": "scale=1920:1080",
            "720p (1280×720)": "scale=1280:720",
            "480p (854×480)": "scale=854:480",
            "360p (640×360)": "scale=640:360"
        }
        cmd += ["-crf", "23", "-vf", res_map.get(resolution, "scale=1280:720")]

    elif method == "Target a max bitrate":
        cmd += ["-crf", str(crf_val), "-maxrate", f"{max_bitrate}k", "-bufsize", f"{max_bitrate*2}k"]

    # Audio mapping
    cmd += ["-c:a", "aac", "-b:a", "128k", output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")


def compress_pdf(input_path: str, output_path: str, strength: int):
    if not PIKEPDF_AVAILABLE:
        raise RuntimeError("pikepdf is not installed. Run: pip install pikepdf")

    jpeg_quality = max(10, 95 - int(strength * 0.8))

    with pikepdf.open(input_path) as pdf:
        if PIL_AVAILABLE and strength >= 30:
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


def compress_image(input_path: str, output_path: str, strength: int):
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is not installed. Run: pip install pillow")

    quality = max(5, 95 - int(strength * 0.85))
    img = Image.open(input_path)
    if img.mode in ("RGBA", "P") and Path(output_path).suffix.lower() in (".jpg", ".jpeg"):
        img = img.convert("RGB")

    save_kwargs = {}
    ext = Path(output_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        save_kwargs = {"quality": quality, "optimize": True}
    elif ext == ".png":
        save_kwargs = {"optimize": True, "compress_level": min(9, 1 + strength // 11)}
    elif ext == ".webp":
        save_kwargs = {"quality": quality}

    img.save(output_path, **save_kwargs)


def compress_generic(input_path: str, output_path: str, strength: int):
    level = max(1, min(9, round(1 + (strength / 100) * 8)))
    with open(input_path, "rb") as f_in, gzip.open(output_path, "wb", compresslevel=level) as f_out:
        shutil.copyfileobj(f_in, f_out)


# --------------------------------------------------------------------------
# Streamlit UI Configuration & Custom Styles
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Zarp Offline — Video Compression Engine",
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
    --signal-blue: #1e5c97;
    --sky-cyan: #4fc3f7;
    --cloud-50: #f7f9fc;
    --cloud-100: #eef2f8;
    --line: #dde4ee;
    --slate-500: #5c6b81;
    --slate-700: #374357;
  }

  .stApp {
    background-color: var(--cloud-50) !important;
    font-family: 'Inter', sans-serif !important;
  }

  h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--navy-950) !important;
  }

  #MainMenu, footer, header { visibility: hidden; }

  /* Custom Top Header */
  .custom-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 40px;
    background: var(--navy-950);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin: -6rem -5rem 0rem -5rem;
  }
  .custom-header img { height: 42px; display: block; }
  
  /* Hero Banner */
  .custom-hero {
    position: relative;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(30,92,151,0.55), transparent 60%), var(--navy-900);
    color: #fff;
    padding: 50px 20px 70px;
    text-align: center;
    margin: 0 -5rem 2rem -5rem;
  }
  .custom-hero h1 {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 10px;
    color: #ffffff !important;
  }
  .custom-hero h1 span {
    background: linear-gradient(100deg, #7fd8ff, #4fc3f7 45%, #8fb8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .custom-hero p {
    color: #a9bcd6;
    font-size: 16px;
    margin: 0 auto;
    max-width: 580px;
  }

  /* Left Sidebar Styling */
  [data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid var(--line) !important;
    padding-top: 2rem !important;
  }

  /* Buttons */
  .stButton>button {
    width: 100%;
    background: linear-gradient(135deg, var(--signal-blue), var(--navy-800)) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 14px 28px !important;
    box-shadow: 0 10px 20px -10px rgba(30,92,151,0.5) !important;
    transition: transform 0.15s ease !important;
  }
  .stButton>button:hover {
    transform: translateY(-2px);
  }

  /* Custom Footer */
  .custom-footer {
    text-align: center;
    padding: 40px 20px 20px;
    color: var(--slate-500);
    font-size: 13.5px;
    border-top: 1px solid var(--line);
    margin-top: 60px;
  }
</style>

<!-- Header Markup -->
<div class="custom-header">
  <div><img src="https://assets.zyrosite.com/cdn-cgi/image/format=auto,w=375,fit=crop/YZ9byglzQnTGj8EP/zarp-labs-logo-jul-2024-AGB4Obj4LkSRGZ5J.png" alt="Zarp Labs"></div>
  <div><img src="https://assets.zyrosite.com/cdn-cgi/image/format=auto,w=768,h=401,fit=crop/YZ9byglzQnTGj8EP/asset-6new-XQNDAvZfAELCSVNr.png" alt="Zarp Offline"></div>
</div>

<!-- Hero Markup -->
<div class="custom-hero">
  <h1>Zarp Offline <span>Video Compression Engine</span></h1>
  <p>Compress videos fast, and without compromise — built for Zarp Offline.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# LEFT SIDEBAR: Video Quality & Size Control Panel
# --------------------------------------------------------------------------

st.sidebar.markdown("### ⚙️ Video Quality & Size")

# 1. Video Codec Options
codec = st.sidebar.selectbox(
    "Video Codec",
    ["H.264 - CPU", "H.264 - GPU 👑 Pro", "H.265 - CPU", "H.265 - GPU 👑 Pro", "AV1 - GPU 👑 Pro"]
)
st.sidebar.caption("H265 codec can reduce video size 20-75% more compared to H264.")

st.sidebar.markdown("---")

# 2. Compression Method Selector
method = st.sidebar.selectbox(
    "Compression Method",
    [
        "Target a file size (Percentage)",
        "Target a file size (MB)",
        "Target a video quality",
        "Target a video resolution",
        "Target a max bitrate"
    ]
)
st.sidebar.caption("Choose 'Target a file size' to get an exact output file size. Choose 'Target a video quality' when quality is important.")

st.sidebar.markdown("---")

# 3. Dynamic Controls in Sidebar
pct_val, mb_val, crf_val, preset_speed, resolution, max_bitrate = 60, 50.0, 21, "Very fast (Default)", "720p (1280×720)", 2000

if method == "Target a file size (Percentage)":
    pct_val = st.sidebar.slider("Select Target Size (%)", min_value=0, max_value=100, value=60)
    st.sidebar.caption("Select target size as a percentage of original.")

elif method == "Target a file size (MB)":
    mb_val = st.sidebar.number_input("Target Size (MB)", min_value=1.0, max_value=10240.0, value=50.0)
    st.sidebar.caption("Enter desired output size in MB.")

elif method == "Target a video quality":
    crf_option = st.sidebar.selectbox(
        "Select Quality (CRF)",
        ["18 Visually lossless - large size", "21 Good quality - medium size (default)", "24 Balanced - smaller size", "28 Compact - noticeable loss"],
        index=1
    )
    crf_val = int(crf_option.split()[0])

    preset_speed = st.sidebar.selectbox(
        "Compression Speed",
        ["Very fast (Default)", "Fast", "Medium", "Slow - best compression"]
    )
    st.sidebar.caption("Slower speeds yield better compression/quality.")

elif method == "Target a video resolution":
    resolution = st.sidebar.selectbox(
        "Select Preset Size",
        ["1080p (1920×1080)", "720p (1280×720)", "480p (854×480)", "360p (640×360)"],
        index=1
    )

elif method == "Target a max bitrate":
    max_bitrate = st.sidebar.number_input("Max Bitrate (Kbps)", min_value=100, max_value=50000, value=2000)

downscale_video = st.sidebar.checkbox("Also downscale video resolution at high strength (≥60%)", value=True)


# --------------------------------------------------------------------------
# MAIN BODY: File Upload & Results Area
# --------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

uploaded_file = st.file_uploader("Upload a file", type=None)

if uploaded_file is not None:
    category = detect_category(uploaded_file.name)
    st.write(f"**Detected type:** {category.capitalize()}  |  **Original size:** {format_bytes(uploaded_file.size)}")

    start = st.button("🚀 Start Compression", type="primary", use_container_width=True)

    if start:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, uploaded_file.name)
            with open(in_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            original_size = os.path.getsize(in_path)
            ext = Path(uploaded_file.name).suffix
            stem = Path(uploaded_file.name).stem

            try:
                with st.spinner("Compressing... this can take a while for large files."):
                    t0 = time.time()

                    if category == "video":
                        if not ffmpeg_available():
                            raise RuntimeError("ffmpeg was not found on this system. Install ffmpeg to compress videos.")
                        out_path = os.path.join(tmpdir, f"{stem}_compressed{ext}")
                        compress_video_advanced(
                            in_path, out_path, codec, method,
                            pct_val, mb_val, crf_val, preset_speed,
                            resolution, max_bitrate, downscale_video
                        )
                        out_name = f"{stem}_compressed{ext}"

                    elif category == "pdf":
                        out_path = os.path.join(tmpdir, f"{stem}_compressed.pdf")
                        compress_pdf(in_path, out_path, pct_val)
                        out_name = f"{stem}_compressed.pdf"

                    elif category == "image":
                        out_path = os.path.join(tmpdir, f"{stem}_compressed{ext}")
                        compress_image(in_path, out_path, pct_val)
                        out_name = f"{stem}_compressed{ext}"

                    else:
                        out_path = os.path.join(tmpdir, f"{stem}{ext}.gz")
                        compress_generic(in_path, out_path, pct_val)
                        out_name = f"{stem}{ext}.gz"

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
                    "strength": pct_val,
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
    st.subheader("📊 Compression Report")

    original = result["original_size"]
    compressed = result["compressed_size"]
    reduction_pct = max(0.0, (1 - compressed / original) * 100) if original else 0.0
    remaining_pct = 100 - reduction_pct

    col1, col2, col3 = st.columns(3)
    col1.metric("Original Size", format_bytes(original))
    col2.metric("Compressed Size", format_bytes(compressed), delta=f"-{reduction_pct:.1f}%")
    col3.metric("Space Saved", format_bytes(max(0, original - compressed)))

    st.progress(min(1.0, remaining_pct / 100), text=f"Final file is {remaining_pct:.1f}% of original size")

    st.bar_chart({"Size (bytes)": {"Original": original, "Compressed": compressed}})

    st.write(
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
# Footer
# --------------------------------------------------------------------------

st.markdown("""
<div class="custom-footer">
  <div>Built with <span style="color:#e63946;">♥</span> by <b>Zarp Labs R&D</b></div>
  <div style="margin-top: 6px; color:#9aa7bb;">© Zarp Offline — Empowering the Last Mile</div>
</div>
""", unsafe_allow_html=True)