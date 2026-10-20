import streamlit as st
import streamlit.components.v1 as components
import os
import cv2
import numpy as np
import base64
import requests
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Streamlit config
st.set_page_config(page_title="Celebrity Recognition (Gemini AI)", page_icon="🌟", layout="wide")

# Create upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Declare components
pause_detector_path = os.path.join(os.path.dirname(__file__), "components", "pause_detector")
pause_detector_button = None
if os.path.exists(pause_detector_path):
    try:
        pause_detector_button = components.declare_component("pause_detector_button", path=pause_detector_path)
    except Exception as e:
        pause_detector_button = None

live_tv_player_path = os.path.join(os.path.dirname(__file__), "components", "live_tv_player")
live_tv_player = None
if os.path.exists(live_tv_player_path):
    try:
        live_tv_player = components.declare_component("live_tv_player", path=live_tv_player_path)
    except Exception as e:
        live_tv_player = None

# Sidebar - Gemini Configuration
st.sidebar.title("⚙️ AI Configuration")
env_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
user_gemini_key = st.sidebar.text_input(
    "Google Gemini API Key",
    value=st.session_state.get("gemini_key", env_gemini_key),
    type="password",
    help="Enter your Google Gemini API key or set GEMINI_API_KEY in .env"
)
if user_gemini_key:
    st.session_state["gemini_key"] = user_gemini_key.strip()
    GEMINI_API_KEY = user_gemini_key.strip()
    st.sidebar.success("🟢 Gemini API Ready")
else:
    GEMINI_API_KEY = ""
    st.sidebar.warning("⚠️ Enter Gemini API key above or in .env")

gemini_model = st.sidebar.selectbox(
    "Gemini Model",
    ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"],
    index=0
)

# Face Detection Functions
def detect_faces_from_rgb(img_rgb):
    """Detects frontal faces in an RGB image using OpenCV Haar cascade."""
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(40, 40))
        if len(faces) == 0:
            return []
        return faces
    except Exception as e:
        print(f"[Face Detection Error] {e}")
        return []

def draw_faces(image, faces, celebrity_name):
    """Draws bounding boxes and labels around detected faces."""
    annotated = image.copy()
    display_label = celebrity_name if celebrity_name and celebrity_name != "Unknown" else "Detected Person"
    
    for (x, y, w, h) in faces:
        # Draw box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 120), 3)
        
        # Label background
        label_size, _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        top_y = max(y - 12, label_size[1] + 10)
        cv2.rectangle(annotated, (x, top_y - label_size[1] - 8), (x + label_size[0] + 12, top_y + 4), (0, 200, 100), -1)
        cv2.putText(annotated, display_label, (x + 6, top_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
        
    return annotated

def extract_frame_at_timestamp(video_path, timestamp_seconds):
    """Extracts a video frame at a given timestamp in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, []
    
    fps = max(1.0, float(cap.get(cv2.CAP_PROP_FPS) or 25.0))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame = int(timestamp_seconds * fps)
    
    if total_frames > 0:
        target_frame = min(max(0, target_frame), total_frames - 1)
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None, []
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detect_faces_from_rgb(frame_rgb)
    return frame_rgb, faces

def extract_name(content):
    """Parses the real celebrity / person full name from Gemini response."""
    if not content or content.strip() in ["Unknown", ""]:
        return "Unknown"
    for line in content.splitlines():
        clean = line.strip()
        if clean.lower().startswith("- **full name**:") or clean.lower().startswith("**full name**:"):
            parts = clean.split(":", 1)
            if len(parts) > 1:
                name = parts[1].strip(" *\"'()[]")
                if name and name.lower() not in ["unknown", "n/a", "none"]:
                    return name
        elif clean.lower().startswith("full name:"):
            parts = clean.split(":", 1)
            if len(parts) > 1:
                name = parts[1].strip(" *\"'()[]")
                if name and name.lower() not in ["unknown", "n/a", "none"]:
                    return name
    return "Unknown"

def get_info(image_data, api_key=None, model=None):
    """Sends image to Gemini API to recognize the celebrity or public figure."""
    # Convert image_data into base64
    if isinstance(image_data, str) and os.path.exists(image_data):
        with open(image_data, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
    elif isinstance(image_data, (bytes, bytearray)):
        img_base64 = base64.b64encode(image_data).decode()
    elif isinstance(image_data, np.ndarray):
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode()
    else:
        return "Unknown"

    prompt_text = """You are an expert AI trained in recognizing celebrities, cinema actors, actresses, and famous public figures with high accuracy.
Carefully analyze the given image and identify the person with certainty.
If a well-known person is present, return the details in this exact format:

- **Full Name**: <Person's Real Full Name>
- **Profession**: <Profession, e.g. Actor, Actress, Singer, Athlete>
- **Nationality**: <Country of origin>
- **Famous For**: <Notable movies, series, or achievements, max 2-3>

If multiple people are detected, return only the most famous one.
If no famous person is recognized, respond strictly with "Unknown".
"""

    key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    req_model = model or gemini_model or "gemini-3.8-flash"
    
    if key:
        models_to_try = [req_model, "gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        for mod in models_to_try:
            try:
                clean_mod = mod.replace("models/", "")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_mod}:generateContent?key={key}"
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": prompt_text},
                            {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}
                        ]
                    }],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 1024
                    }
                }
                response = requests.post(url, json=payload, timeout=30.0)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            ans = parts[0]["text"].strip()
                            if ans:
                                return ans
                elif response.status_code in [404, 503, 429]:
                    print(f"[Gemini Notice] Model {clean_mod} returned HTTP {response.status_code}, trying next model...")
                    continue
                else:
                    print(f"[Gemini Error] HTTP {response.status_code}: {response.text[:200]}")
            except Exception as e:
                print(f"[Gemini Error on {clean_mod}] {e}, trying next model...")
                continue

    # Fallback to Groq if configured
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        payload = {
            "model": "qwen/qwen3.8-27b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20.0)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
        except Exception as ge:
            print(f"[Groq Error] {ge}")

    return "Unknown"


# ==========================================
# STREAMLIT UI - VIDEO & LIVE STREAM DETECTION
# ==========================================

st.title("🌟 Celebrity Video & Live Stream Recognition")
st.markdown("Play live TV or any video, click **Detect**, and identify celebrities in real-time with **Google Gemini AI**.")

source_options = [
    "📡 Live TV Stream (M3U8)",
    "📤 Upload Video / Movie File"
]
video_source_type = st.radio("Video Source:", source_options, horizontal=True)

active_video_path = None
is_live_stream = False

if video_source_type == "📡 Live TV Stream (M3U8)":
    is_live_stream = True
    live_stream_url = st.text_input(
        "Live TV M3U8 Stream URL:", 
        value="https://ghost2.uk/live/JohnHyndCC/hzd2eujtq7/2451936.m3u8"
    )
    active_video_path = live_stream_url.strip() if live_stream_url else None
elif video_source_type == "📤 Upload Video / Movie File":
    uploaded_video = st.file_uploader("Upload a movie / video file (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        active_video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
        with open(active_video_path, "wb") as vf:
            vf.write(uploaded_video.getbuffer())

# 1. LIVE TV STREAMING
if is_live_stream and active_video_path:
    st.markdown(f"🔴 **Live TV Broadcast:** `{active_video_path}`")
    
    # Render Live TV Player with frame capture on pause time
    if live_tv_player:
        live_data = live_tv_player(stream_url=active_video_path, key="live_tv_hls_comp")
        if live_data and isinstance(live_data, dict):
            nonce = live_data.get("nonce")
            if nonce and nonce != st.session_state.get("last_live_nonce"):
                st.session_state["last_live_nonce"] = nonce
                img_b64 = live_data.get("image_base64", "")
                p_time = float(live_data.get("paused_time", 0.0))
                
                with st.spinner(f"📸 Extracting frame from paused moment ({p_time:.1f}s) & identifying with Gemini AI..."):
                    f_rgb = None
                    
                    # 1. Primary: Exact frame captured on pause from browser video canvas
                    if img_b64 and "," in img_b64:
                        try:
                            raw_b64 = img_b64.split(",", 1)[1]
                            nparr = np.frombuffer(base64.b64decode(raw_b64), np.uint8)
                            f_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if f_bgr is not None:
                                f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
                        except Exception as decode_err:
                            print(f"[Live Canvas Decode Error] {decode_err}")
                    
                    # 2. Fallback: If canvas was tainted, capture from OpenCV
                    if f_rgb is None:
                        cap = cv2.VideoCapture(active_video_path)
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None:
                            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if f_rgb is not None:
                        f_faces = detect_faces_from_rgb(f_rgb)
                        info = get_info(f_rgb, api_key=GEMINI_API_KEY, model=gemini_model)
                        p_name = extract_name(info)

                        st.session_state["v_info"] = info
                        st.session_state["v_name"] = p_name
                        st.session_state["v_frame"] = f_rgb
                        st.session_state["v_faces"] = f_faces
                        st.session_state["v_ts"] = p_time
                        st.rerun()
                    else:
                        st.error("⚠️ Could not capture frame from stream. Please retry.")
    else:
        st.warning("Live TV Player component not initialized.")

# 2. LOCAL MOVIE / VIDEO PLAYBACK
elif (not is_live_stream) and active_video_path and os.path.exists(active_video_path):
    cap = cv2.VideoCapture(active_video_path)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = max(1.0, float(cap.get(cv2.CAP_PROP_FPS) or 25.0))
    duration = tot_frames / fps if tot_frames > 0 else 0
    cap.release()

    mins = int(duration // 60)
    secs = int(duration % 60)
    st.info(f"📹 **Video:** `{os.path.basename(active_video_path)}` | ⏱️ **Duration:** `{mins:02d}:{secs:02d}` ({duration:.1f}s)")

    # Render Video Player
    st.video(active_video_path)

    # Instructions banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(236, 72, 153, 0.15) 100%); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 12px; padding: 12px 18px; text-align: center; margin: 12px 0 16px 0;'>
        <span style='color: #a5b4fc; font-weight: 800;'>STEP 1:</span> Play or pause video when the person is on screen &nbsp;•&nbsp; 
        <span style='color: #f472b6; font-weight: 800;'>STEP 2:</span> Click the glowing button below to identify them with Gemini!
    </div>
    """, unsafe_allow_html=True)

    if pause_detector_button:
        pause_data = pause_detector_button(key="video_live_detector")
        if pause_data and isinstance(pause_data, dict):
            nonce = pause_data.get("nonce")
            if nonce and nonce != st.session_state.get("last_video_nonce"):
                st.session_state["last_video_nonce"] = nonce
                ts = float(pause_data.get("timestamp", 0.0))
                img_b64 = pause_data.get("image_base64", "")
                
                with st.spinner(f"📸 Extracting frame at {ts:.2f}s and recognizing with Gemini AI..."):
                    f_rgb, f_faces = extract_frame_at_timestamp(active_video_path, ts)
                    
                    if f_rgb is None and img_b64 and "," in img_b64:
                        raw_b64 = img_b64.split(",", 1)[1]
                        nparr = np.frombuffer(base64.b64decode(raw_b64), np.uint8)
                        f_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if f_bgr is not None:
                            f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
                            f_faces = detect_faces_from_rgb(f_rgb)

                    if f_rgb is not None:
                        info = get_info(f_rgb, api_key=GEMINI_API_KEY, model=gemini_model)
                        p_name = extract_name(info)
                        
                        st.session_state["v_info"] = info
                        st.session_state["v_name"] = p_name
                        st.session_state["v_frame"] = f_rgb
                        st.session_state["v_faces"] = f_faces
                        st.session_state["v_ts"] = ts
                        st.rerun()

    # Alternative Manual Scrubber
    with st.expander("⏱️ Manual Frame Capture / Timestamp Scrubber"):
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            scrub_ts = st.slider("Select timestamp (seconds):", 0.0, float(max(1.0, duration)), 0.0, 0.5)
        with col_s2:
            st.write("")
            st.write("")
            if st.button("🎯 Detect at Timestamp", use_container_width=True):
                with st.spinner(f"Extracting frame at {scrub_ts:.1f}s..."):
                    f_rgb, f_faces = extract_frame_at_timestamp(active_video_path, scrub_ts)
                    if f_rgb is not None:
                        info = get_info(f_rgb, api_key=GEMINI_API_KEY, model=gemini_model)
                        p_name = extract_name(info)
                        
                        st.session_state["v_info"] = info
                        st.session_state["v_name"] = p_name
                        st.session_state["v_frame"] = f_rgb
                        st.session_state["v_faces"] = f_faces
                        st.session_state["v_ts"] = scrub_ts
                        st.rerun()

# 3. DISPLAY IDENTIFIED CELEBRITY RESULTS
if st.session_state.get("v_name"):
    v_name = st.session_state["v_name"]
    v_info = st.session_state["v_info"]
    v_frame = st.session_state["v_frame"]
    v_faces = st.session_state.get("v_faces", [])
    v_ts = st.session_state.get("v_ts", 0.0)

    st.markdown("---")
    if v_name != "Unknown":
        st.success(f"### 👤 Person Identified: **{v_name}**")
    else:
        st.warning("### 👤 Person: Unknown / Unrecognized")

    col_res1, col_res2 = st.columns([1, 1])
    with col_res1:
        if len(v_faces) > 0:
            annotated_frame = draw_faces(v_frame, v_faces, v_name)
            st.image(annotated_frame, caption=f"Detected Face at {v_ts:.1f}s", use_container_width=True)
        else:
            st.image(v_frame, caption=f"Captured Frame at {v_ts:.1f}s", use_container_width=True)

    with col_res2:
        st.markdown("#### 📋 Gemini Intelligence Dossier")
        st.markdown(v_info)
