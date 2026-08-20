"""
Universal Data Compressor - Streamlit App
------------------------------------------
Upload video, PDF, images, or any other file, choose a compression
strength, and get a compressed output plus a dashboard showing how
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


# ---- Compression strategies -----------------------------------------------

def compress_video(input_path: str, output_path: str, strength: int, downscale: bool):
    """
    Map compression strength (0-100) to an H.264 CRF value.
    Higher strength -> higher CRF -> smaller file, lower quality.
    CRF range used: 18 (near-lossless) .. 40 (heavy compression)
    """
    crf = int(18 + (strength / 100) * 22)
    audio_bitrate = "128k" if strength < 50 else "96k" if strength < 80 else "64k"

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
        "-c:a", "aac", "-b:a", audio_bitrate,
    ]

    if downscale and strength >= 60:
        # Scale down resolution for heavier compression settings
        cmd += ["-vf", "scale=trunc(iw*0.7/2)*2:trunc(ih*0.7/2)*2"]

    cmd += [output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")


def compress_pdf(input_path: str, output_path: str, strength: int):
    """
    Recompress a PDF using pikepdf: recompress streams and, at higher
    strengths, downsample/re-encode embedded images more aggressively.
    """
    if not PIKEPDF_AVAILABLE:
        raise RuntimeError(
            "pikepdf is not installed. Run: pip install pikepdf"
        )

    jpeg_quality = max(10, 95 - int(strength * 0.8))  # 95 -> 15 as strength rises

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
                        # Skip images that can't be re-encoded (e.g. masks, CMYK edge cases)
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
    """Lossless gzip compression, level scaled by strength (1-9)."""
    level = max(1, min(9, round(1 + (strength / 100) * 8)))
    with open(input_path, "rb") as f_in, gzip.open(output_path, "wb", compresslevel=level) as f_out:
        shutil.copyfileobj(f_in, f_out)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Zarp Offline Video Compression Enginer", page_icon="🗜️", layout="centered")

st.title("🗜️ Zarp Offline Video Compression Enginer")
st.caption("Upload video, PDF, images, or any other file. Choose a compression level and compress it.")

if "result" not in st.session_state:
    st.session_state.result = None

uploaded_file = st.file_uploader("Upload a file", type=None)

st.sidebar.header("⚙️ Compression Settings")
strength = st.sidebar.slider(
    "Compression strength (%)",
    min_value=0, max_value=100, value=60, step=5,
    help="Higher = smaller file, lower quality.",
)

st.sidebar.subheader("Advanced")
downscale_video = st.sidebar.checkbox(
    "Also downscale video resolution at high strength (≥60%)", value=True
)
st.sidebar.caption(
    "Video needs ffmpeg installed on this machine. "
    "PDF compression needs the `pikepdf` (and optionally `pillow`) package."
)

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
                with st.spinner("Compressing... this can take a while for large videos."):
                    t0 = time.time()

                    if category == "video":
                        if not ffmpeg_available():
                            raise RuntimeError(
                                "ffmpeg was not found on this system. Install ffmpeg to compress videos."
                            )
                        out_path = os.path.join(tmpdir, f"{stem}_compressed{ext}")
                        compress_video(in_path, out_path, strength, downscale_video)
                        out_name = f"{stem}_compressed{ext}"

                    elif category == "pdf":
                        out_path = os.path.join(tmpdir, f"{stem}_compressed.pdf")
                        compress_pdf(in_path, out_path, strength)
                        out_name = f"{stem}_compressed.pdf"

                    elif category == "image":
                        out_path = os.path.join(tmpdir, f"{stem}_compressed{ext}")
                        compress_image(in_path, out_path, strength)
                        out_name = f"{stem}_compressed{ext}"

                    else:
                        out_path = os.path.join(tmpdir, f"{stem}{ext}.gz")
                        compress_generic(in_path, out_path, strength)
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
                    "strength": strength,
                }
                st.success(f"Compression complete in {elapsed:.1f}s ✅")

            except Exception as e:
                st.session_state.result = None
                st.error(f"Compression failed: {e}")

# --------------------------------------------------------------------------
# Dashboard
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

    st.bar_chart(
        {"Size (bytes)": {"Original": original, "Compressed": compressed}},
    )

    st.write(
        f"Requested compression strength: **{result['strength']}%** &nbsp;|&nbsp; "
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
