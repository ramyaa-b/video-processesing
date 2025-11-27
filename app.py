# app.py
"""
Streamlit app: Video -> Frames (NO transcription, no ffmpeg)
Features:
- Upload video files
- Choose a saved video from dropdown
- Extract frames (OpenCV) and preview a few frames
- Download extracted frames as a zip
Storage: .temp_streamlit/ in working directory
"""

from pathlib import Path
import zipfile
import shutil
import io
import tempfile

import streamlit as st

from PIL import Image

BASE_DIR = Path.cwd() / ".temp_streamlit"
VIDEOS_DIR = BASE_DIR / "videos"
FRAMES_DIR = BASE_DIR / "frames"

for d in (BASE_DIR, VIDEOS_DIR, FRAMES_DIR):
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Video → Frames (no ffmpeg)", layout="wide")
st.title("📼 Video → Frames — *no transcription*")

# --- Helpers ---
def save_uploaded_video(uploaded_file):
    target = VIDEOS_DIR / uploaded_file.name
    # avoid overwrite
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        i = 1
        while (VIDEOS_DIR / f"{stem}_{i}{suffix}").exists():
            i += 1
        target = VIDEOS_DIR / f"{stem}_{i}{suffix}"
    with target.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return target

def list_saved_videos():
    return sorted([p.name for p in VIDEOS_DIR.iterdir() if p.is_file()])

def extract_frames_opencv(video_path: Path, out_dir: Path, frame_step: int = 1, max_preview=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open the video. Unsupported codec or corrupted file.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    idx = 0
    saved = 0
    preview_imgs = []
    pbar = st.progress(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            fname = out_dir / f"frame_{saved:06d}.jpg"
            # write as JPEG
            cv2.imwrite(str(fname), frame)
            if len(preview_imgs) < max_preview:
                # convert BGR->RGB for display
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_imgs.append(Image.fromarray(img_rgb))
            saved += 1
        idx += 1
        if total:
            pbar.progress(min(1.0, idx / total))
    pbar.empty()
    cap.release()
    return {"saved": saved, "fps": fps, "total": total, "previews": preview_imgs}

def make_zip_bytes(folder: Path, base_name: str = "frames"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(folder.iterdir()):
            if f.is_file():
                # arcname: relative filename inside the zip
                zf.write(f, arcname=f.name)
    buf.seek(0)
    return buf

# --- UI ---
st.sidebar.header("Upload / Manage")
uploaded = st.sidebar.file_uploader("Upload a video (mp4, avi, mov, mkv...)", type=["mp4","mov","mkv","avi"], accept_multiple_files=False)
if uploaded:
    saved = save_uploaded_video(uploaded)
    st.sidebar.success(f"Saved: {saved.name}")

saved_videos = ["-- pick one --"] + list_saved_videos()
chosen = st.selectbox("Choose a saved video", options=saved_videos)

col1, col2 = st.columns([3,1])

with col1:
    st.markdown("### Video preview / info")
    if chosen and chosen != "-- pick one --":
        video_path = VIDEOS_DIR / chosen
        try:
            st.video(str(video_path))
        except Exception:
            st.info("Streamlit can't preview this video; it may be too large or use an unsupported codec.")
        st.write(f"**Path:** `{str(video_path)}`")
    else:
        st.info("No video selected. Upload a video from the sidebar or select one from the dropdown.")

with col2:
    st.markdown("### Actions")
    frame_step = st.number_input("Frame step (1 = every frame, 30 = every 30th frame)", min_value=1, step=1, value=1)
    process_btn = st.button("Process to frames")
    clear_btn = st.button("Clear all saved videos & frames (danger)")

if clear_btn:
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    for d in (BASE_DIR, VIDEOS_DIR, FRAMES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    st.experimental_rerun()

if process_btn:
    if not chosen or chosen == "-- pick one --":
        st.warning("Choose a saved video first.")
    else:
        video_path = VIDEOS_DIR / chosen
        out_folder = FRAMES_DIR / video_path.stem
        st.info(f"Extracting frames to `{out_folder}` (this may take a while for long videos)")
        try:
            res = extract_frames_opencv(video_path, out_folder, frame_step=int(frame_step))
            st.success(f"Saved {res['saved']} frames (reported total_frames={res['total']}, fps={res['fps']:.2f})")
            if res["previews"]:
                st.markdown("#### Frame previews")
                for i, img in enumerate(res["previews"]):
                    st.image(img, caption=f"Preview {i+1}", use_column_width=True)
            # provide ZIP download
            if res["saved"] > 0:
                zip_buf = make_zip_bytes(out_folder, base_name=video_path.stem + "_frames")
                st.download_button(
                    label="Download all frames (.zip)",
                    data=zip_buf,
                    file_name=f"{video_path.stem}_frames.zip",
                    mime="application/zip"
                )
        except Exception as e:
            st.error(f"Frame extraction failed: {e}")

st.markdown("---")
with st.expander("Saved videos (debug)"):
    for v in list_saved_videos():
        st.write(v)
