# app.py
from pathlib import Path
import zipfile
import shutil
import io

import streamlit as st
import cv2
from PIL import Image

BASE_DIR = Path.cwd() / ".temp_streamlit"
VIDEOS_DIR = BASE_DIR / "videos"
FRAMES_DIR = BASE_DIR / "frames"

for d in (BASE_DIR, VIDEOS_DIR, FRAMES_DIR):
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Video → Frames (no ffmpeg)", layout="wide")
st.title("📼 Video → Frames — *no transcription*")

# Initialize session_state entry for selected video if missing
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "last_saved_name" not in st.session_state:
    st.session_state.last_saved_name = None

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
                zf.write(f, arcname=f.name)
    buf.seek(0)
    return buf

# --- UI ---
st.sidebar.header("Upload / Manage")
uploaded = st.sidebar.file_uploader("Upload a video (mp4, avi, mov, mkv...)", type=["mp4","mov","mkv","avi"], accept_multiple_files=False)
if uploaded:
    saved = save_uploaded_video(uploaded)
    st.session_state.last_saved_name = saved.name      # store actual saved filename
    st.session_state.selected_video = saved.name       # auto-select it
    st.sidebar.success(f"Saved: {saved.name}")

# Build the list AFTER potential save so it includes the new file
saved_videos = ["-- pick one --"] + list_saved_videos()

# Use selectbox with a key bound to session_state.selected_video
chosen = st.selectbox("Choose a saved video", options=saved_videos, index=0, key="selected_video")

# If session_state was set to a file that exists, ensure selectbox shows it:
# (Streamlit will handle this because key="selected_video" is bound)

col1, col2 = st.columns([3,1])

with col1:
    st.markdown("### Video preview / info")
    if st.session_state.selected_video and st.session_state.selected_video != "-- pick one --":
        video_path = VIDEOS_DIR / st.session_state.selected_video
        if video_path.exists():
            try:
                # reliably feed raw bytes to st.video
                video_bytes = video_path.read_bytes()
                st.video(video_bytes)
            except Exception:
                st.info("Streamlit can't preview this video; it may be too large or use an unsupported codec.")
            st.write(f"**Path:** `{str(video_path)}`")
        else:
            st.error(f"Selected file not found on disk: {st.session_state.selected_video}")
            # show what's actually on disk for debugging
            st.write("Files on disk:")
            for p in list_saved_videos():
                st.write("-", p)
    else:
        if st.session_state.last_saved_name:
            st.info(f"Last saved file: `{st.session_state.last_saved_name}` — it should be selectable in the dropdown.")
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
    st.session_state.selected_video = None
    st.session_state.last_saved_name = None
    st.experimental_rerun()

if process_btn:
    if not st.session_state.selected_video or st.session_state.selected_video == "-- pick one --":
        st.warning("Choose a saved video first.")
    else:
        video_path = VIDEOS_DIR / st.session_state.selected_video
        out_folder = FRAMES_DIR / video_path.stem
        st.info(f"Extracting frames to `{out_folder}` (this may take a while for long videos)")
        try:
            res = extract_frames_opencv(video_path, out_folder, frame_step=int(frame_step))
            st.success(f"Saved {res['saved']} frames (reported total_frames={res['total']}, fps={res['fps']:.2f})")
            if res["previews"]:
                st.markdown("#### Frame previews")
                for i, img in enumerate(res["previews"]):
                    st.image(img, caption=f"Preview {i+1}", use_column_width=True)
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
    files = list_saved_videos()
    if files:
        for v in files:
            st.write(v)
    else:
        st.write("No saved videos found")

