# Universal Data Compressor (Streamlit)

Upload video, PDF, images, or any other file, pick a compression strength,
and download the compressed result along with a before/after report.

## Setup

```bash
pip install -r requirements.txt
```

Videos are compressed with **ffmpeg**, which must be installed separately
and available on your system PATH:

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: download from https://ffmpeg.org/download.html and add it to PATH

PDF and image compression use `pikepdf` and `pillow`, both installed via
`requirements.txt`.

## Run

```bash
streamlit run app.py
```

## How it works

| File type | Method | What "strength" controls |
|---|---|---|
| Video (mp4, mov, avi, mkv, ...) | ffmpeg H.264 re-encode | CRF (quality) 18→40, audio bitrate, optional resolution downscale ≥60% |
| PDF | pikepdf stream recompression + JPEG re-encoding of embedded images | JPEG quality of embedded images |
| Image (jpg, png, webp, ...) | Pillow re-save | JPEG/WebP quality or PNG compression level |
| Anything else | gzip | gzip compression level 1–9 |

The **strength slider (0–100%)** is a compression-effort setting, not a
guaranteed output size — actual space saved depends on the file's content
(a video that's already highly compressed will shrink less than raw
footage). The dashboard reports the **actual** achieved reduction after
compression finishes.

## Notes / limitations

- Generic (gzip) compression is lossless; video/image/PDF compression is
  lossy at higher strengths (quality trade-off).
- Very large video files can take a while — ffmpeg re-encoding runs
  synchronously in this simple app.
- For a target *file size* (rather than a quality/strength dial), you'd
  need an extra pass that measures output and retries with adjusted CRF —
  happy to add that if useful.
