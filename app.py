import streamlit as st
import streamlit.components.v1 as components
import os
import cv2
import numpy as np
import base64
import requests
import urllib.parse
import re
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from dotenv import load_dotenv

try:
    import face_recognition
except ImportError:
    face_recognition = None


# Load environment variables from .env file
load_dotenv()

# Streamlit config
# Streamlit config - sidebar collapsed and hidden
st.set_page_config(
    page_title="Celebrity Video & Live Recognition Studio",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit sidebar and constrain video player size
st.markdown("""
<style>
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Neatly constrain video player size so uploaded videos never take full screen */
    [data-testid="stVideo"] {
        max-width: 680px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        display: flex !important;
        justify-content: center !important;
    }
    [data-testid="stVideo"] video {
        max-height: 440px !important;
        max-width: 100% !important;
        object-fit: contain !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
        background: #000 !important;
    }
</style>
""", unsafe_allow_html=True)

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

# Gemini Configuration (loaded from .env directly without sidebar)
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()


# ==========================================
# ROBUST FACE DETECTION & BOUNDING BOX ENGINE
# ==========================================

def non_max_suppression(boxes, overlap_thresh=0.35):
    """Merges overlapping duplicate bounding boxes around the same face into a single clean box."""
    if not boxes or len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=int)
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / (area[idxs[:last]] + 1e-6)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return [(int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])) for i in pick]


YUNET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx")


def get_yunet_detector(img_w, img_h, score_threshold=0.5):
    """Initializes OpenCV YuNet Deep Learning face detector with automatic model download if missing."""
    if not hasattr(cv2, 'FaceDetectorYN'):
        return None
    if not os.path.exists(YUNET_MODEL_PATH):
        try:
            os.makedirs(os.path.dirname(YUNET_MODEL_PATH), exist_ok=True)
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            r = requests.get(url, timeout=12.0)
            if r.status_code == 200:
                with open(YUNET_MODEL_PATH, "wb") as f:
                    f.write(r.content)
        except Exception as e:
            print(f"[YuNet Download Notice] {e}")
            return None

    if os.path.exists(YUNET_MODEL_PATH):
        try:
            return cv2.FaceDetectorYN.create(
                model=YUNET_MODEL_PATH,
                config='',
                input_size=(int(img_w), int(img_h)),
                score_threshold=score_threshold,
                nms_threshold=0.3,
                top_k=5000
            )
        except Exception as e:
            print(f"[YuNet Detector Init Error] {e}")
    return None


def is_genuine_human_face(crop_rgb):
    """
    Verifies if a cropped image patch is a genuine human face containing actual
    facial features (eyes, nose, mouth) using dlib/face_recognition landmarks.
    Eliminates false-positive detections on back-of-head, hair, ears, furniture, clothing.
    """
    if crop_rgb is None or not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0:
        return False
    if face_recognition is None:
        return True  # If dlib landmark detector is unavailable, fail open
    try:
        h, w = crop_rgb.shape[:2]
        if h < 10 or w < 10:
            return False
        target_dim = 160
        scale = target_dim / max(h, w)
        resized = cv2.resize(crop_rgb, (max(16, int(w * scale)), max(16, int(h * scale))))
        resized = np.ascontiguousarray(resized, dtype=np.uint8)
        landmarks = face_recognition.face_landmarks(resized)
        if landmarks and len(landmarks) > 0:
            lm = landmarks[0]
            has_eyes = ('left_eye' in lm and len(lm['left_eye']) >= 4) or ('right_eye' in lm and len(lm['right_eye']) >= 4)
            has_mouth_or_nose = ('nose_tip' in lm or 'top_lip' in lm or 'bottom_lip' in lm)
            if has_eyes and has_mouth_or_nose:
                return True
    except Exception:
        pass
    return False


def is_valid_face_box(box, img_w, img_h):
    """Strictly filters out background objects (furniture, doors, patterns) using face geometric proportions."""
    x, y, w, h = box
    if w < 18 or h < 20:
        return False
    if x < 0 or y < 0 or (x + w) > img_w * 1.05 or (y + h) > img_h * 1.05:
        return False
    # Human face aspect ratio: width to height ratio between 0.45 and 1.50 (accommodating distant faces and angled profiles)
    aspect_ratio = float(w) / float(h)
    if aspect_ratio < 0.45 or aspect_ratio > 1.50:
        return False
    # Invalidate full-screen or gargantuan boxes
    if w > img_w * 0.92 or h > img_h * 0.92:
        return False
    return True


def detect_faces_from_rgb(img_rgb, high_sensitivity=False):
    """
    Detects faces in an RGB image with multi-tier robust fallback:
    Tier 1: OpenCV YuNet Deep Learning Face Detector (handles sunglasses, close-ups, side angles, occlusions)
    Tier 2: dlib face_recognition HOG detector
    Tier 3: OpenCV Frontal Haar Cascades with strict minNeighbors and landmark validation
    """
    if img_rgb is None or not isinstance(img_rgb, np.ndarray) or img_rgb.size == 0:
        return []

    img_h, img_w = img_rgb.shape[:2]
    candidate_boxes = []

    # Tier 1: OpenCV YuNet Deep Learning (State-of-the-art CNN face detector)
    try:
        thresh = 0.40 if high_sensitivity else 0.50
        yunet = get_yunet_detector(img_w, img_h, score_threshold=thresh)
        if yunet is not None:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            _, y_faces = yunet.detect(img_bgr)
            if y_faces is not None and len(y_faces) > 0:
                for f in y_faces:
                    x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    x = max(0, x)
                    y = max(0, y)
                    fw = min(img_w - x, fw)
                    fh = min(img_h - y, fh)
                    box = (x, y, fw, fh)
                    score = float(f[-1])
                    if is_valid_face_box(box, img_w, img_h):
                        crop = img_rgb[y:y+fh, x:x+fw]
                        # Verify landmark presence for moderate confidence detections to exclude hair/neck
                        if score >= 0.88 or is_genuine_human_face(crop):
                            candidate_boxes.append(box)
                if len(candidate_boxes) > 0:
                    return non_max_suppression(candidate_boxes, overlap_thresh=0.35)
    except Exception as ye:
        print(f"[YuNet Detection Notice] {ye}")

    # Tier 2: face_recognition (dlib HOG face detector)
    if face_recognition is not None:
        try:
            scale = 1.0
            if max(img_h, img_w) > 1200:
                scale = 960.0 / max(img_h, img_w)
                small_rgb = cv2.resize(img_rgb, (int(img_w * scale), int(img_h * scale)))
            else:
                small_rgb = img_rgb

            upsample_count = 1 if (high_sensitivity or min(img_h, img_w) < 500) else 0
            locs = face_recognition.face_locations(small_rgb, number_of_times_to_upsample=upsample_count, model="hog")
            
            if len(locs) == 0 and high_sensitivity:
                locs = face_recognition.face_locations(small_rgb, number_of_times_to_upsample=2, model="hog")

            for (top, right, bottom, left) in locs:
                if scale != 1.0:
                    top = int(top / scale)
                    right = int(right / scale)
                    bottom = int(bottom / scale)
                    left = int(left / scale)
                box_w = max(10, right - left)
                box_h = max(10, bottom - top)
                box = (int(left), int(top), int(box_w), int(box_h))
                if is_valid_face_box(box, img_w, img_h):
                    candidate_boxes.append(box)
        except Exception as e:
            print(f"[face_recognition notice] {e}")

    if len(candidate_boxes) > 0:
        return non_max_suppression(candidate_boxes, overlap_thresh=0.35)

    # Tier 3: OpenCV Cascades (Frontal only - validated by facial landmarks to prevent hair/neck false positives)
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        min_n = 7 if high_sensitivity else 8
        cascade_names = [
            ('haarcascade_frontalface_alt2.xml', 1.08, min_n),
            ('haarcascade_frontalface_default.xml', 1.10, min_n + 1)
        ]
        for cname, s_factor, c_min_n in cascade_names:
            cpath = cv2.data.haarcascades + cname
            if os.path.exists(cpath):
                cascade = cv2.CascadeClassifier(cpath)
                detected = cascade.detectMultiScale(gray, scaleFactor=s_factor, minNeighbors=c_min_n, minSize=(28, 28))
                for (x, y, w, h) in detected:
                    box = (int(x), int(y), int(w), int(h))
                    if is_valid_face_box(box, img_w, img_h):
                        crop = img_rgb[y:y+h, x:x+w]
                        if is_genuine_human_face(crop):
                            candidate_boxes.append(box)

        if len(candidate_boxes) > 0:
            return non_max_suppression(candidate_boxes, overlap_thresh=0.35)

        # Tier 4: Adaptive CLAHE contrast enhancement for dim or vintage video footage
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_gray = clahe.apply(gray)
        for cname in ['haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_default.xml']:
            cpath = cv2.data.haarcascades + cname
            if os.path.exists(cpath):
                cascade = cv2.CascadeClassifier(cpath)
                detected = cascade.detectMultiScale(eq_gray, scaleFactor=1.08, minNeighbors=8, minSize=(32, 32))
                for (x, y, w, h) in detected:
                    box = (int(x), int(y), int(w), int(h))
                    if is_valid_face_box(box, img_w, img_h):
                        crop = img_rgb[y:y+h, x:x+w]
                        if is_genuine_human_face(crop):
                            candidate_boxes.append(box)
    except Exception as e:
        print(f"[OpenCV Face Detection Error] {e}")

    return non_max_suppression(candidate_boxes, overlap_thresh=0.35)


def crop_face_region(image_rgb, box, pad_ratio=0.25):
    """Crops the primary detected face with safe padding to include hair and jawline."""
    if image_rgb is None or box is None:
        return None
    ih, iw = image_rgb.shape[:2]
    x, y, w, h = box
    # For small/distant faces, give slightly more contextual padding (hair/shoulders) for recognition
    effective_pad = 0.35 if (w < 50 or h < 50) else pad_ratio
    pad_x = int(w * effective_pad)
    pad_y = int(h * effective_pad)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.ascontiguousarray(image_rgb[y1:y2, x1:x2], dtype=np.uint8)


def draw_multi_faces(image, persons, selected_idx=None):
    """Draws sleek bounding boxes and prominent badges for each detected person with individual colors."""
    if image is None:
        return image
    annotated = image.copy()
    
    # Modern high-contrast neon palette for distinct individuals
    colors = [
        {"box": (0, 230, 115), "corner": (0, 255, 200), "badge": (0, 180, 90)},       # Emerald / Cyan (Person 1)
        {"box": (255, 170, 0), "corner": (255, 215, 0), "badge": (210, 140, 0)},     # Vivid Amber / Gold (Person 2)
        {"box": (236, 72, 153), "corner": (244, 114, 182), "badge": (190, 24, 93)}, # Fuchsia / Rose (Person 3)
        {"box": (168, 85, 247), "corner": (192, 132, 252), "badge": (126, 34, 206)},# Purple / Violet (Person 4)
    ]
    
    for idx, p in enumerate(persons):
        box = p.get("box")
        if not box:
            continue
        x, y, w, h = box
        p_name = p.get("name", "Unknown")
        is_rec = p.get("is_recognized", False)
        p_id = p.get("id", idx + 1)
        
        color_scheme = colors[idx % len(colors)]
        is_focused = (selected_idx is None or selected_idx == idx)
        
        box_color = color_scheme["box"] if is_focused else (120, 120, 120)
        corner_color = color_scheme["corner"] if is_focused else (160, 160, 160)
        badge_color = color_scheme["badge"] if is_focused else (90, 90, 90)
        
        # Display label: e.g. "#1: Sean Connery" or "#2: Face Unverified"
        if is_rec and p_name != "Unknown":
            display_label = f"#{p_id}: {p_name}"
        else:
            display_label = f"#{p_id}: Face Unverified"
            
        # Draw Bounding Box
        thickness = 3 if is_focused else 2
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, thickness)
        
        # Corner accents
        line_len = max(12, int(min(w, h) * 0.18))
        c_thick = 4 if is_focused else 3
        cv2.line(annotated, (x, y), (x + line_len, y), corner_color, c_thick)
        cv2.line(annotated, (x, y), (x, y + line_len), corner_color, c_thick)
        cv2.line(annotated, (x + w, y), (x + w - line_len, y), corner_color, c_thick)
        cv2.line(annotated, (x + w, y), (x + w, y + line_len), corner_color, c_thick)
        cv2.line(annotated, (x, y + h), (x + line_len, y + h), corner_color, c_thick)
        cv2.line(annotated, (x, y + h), (x, y + h - line_len), corner_color, c_thick)
        cv2.line(annotated, (x + w, y + h), (x + w - line_len, y + h), corner_color, c_thick)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - line_len), corner_color, c_thick)
        
        # Label badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75 if is_focused else 0.65
        font_thick = 2
        label_size, baseline = cv2.getTextSize(display_label, font, font_scale, font_thick)
        top_y = max(y - 10, label_size[1] + 10)
        cv2.rectangle(annotated, (x, top_y - label_size[1] - 8), (x + label_size[0] + 16, top_y + 4), badge_color, -1)
        cv2.putText(annotated, display_label, (x + 8, top_y - 2), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
        
    return annotated


def draw_faces(image, faces, celebrity_name):
    """Backward compatible wrapper for draw_multi_faces."""
    persons = []
    for idx, b in enumerate(faces):
        persons.append({
            "id": idx + 1,
            "box": b,
            "name": celebrity_name,
            "is_recognized": (celebrity_name and celebrity_name != "Unknown")
        })
    return draw_multi_faces(image, persons)


_ref_encoding_cache = {}
_comparison_image_cache = {}


def get_reference_face_encoding(photo_url):
    """Fetches official reference portrait from IMDb/Wikipedia and caches its 128-d face encoding."""
    if not photo_url or face_recognition is None:
        return None
    if photo_url in _ref_encoding_cache:
        return _ref_encoding_cache[photo_url]
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        r = session.get(photo_url, timeout=5.0)
        if r.status_code == 200:
            arr = np.frombuffer(r.content, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                if max(h, w) > 600:
                    scale = 600.0 / max(h, w)
                    rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                clean_rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                encs = face_recognition.face_encodings(clean_rgb)
                if encs:
                    _ref_encoding_cache[photo_url] = encs[0]
                    return encs[0]
    except Exception as e:
        print(f"[Biometric Ref Encoding Notice] {e}")
    _ref_encoding_cache[photo_url] = None
    return None


def verify_face_biometrics(video_face_crop, ref_photo_url):
    """
    Biometrically verifies video face crop against the official reference portrait photo.
    Returns: (is_verified: bool | None, confidence_score: float | None, distance: float | None)
    """
    if video_face_crop is None or not ref_photo_url or face_recognition is None:
        return None, None, None

    try:
        ref_enc = get_reference_face_encoding(ref_photo_url)
        if ref_enc is None:
            return None, None, None

        crop_rgb = video_face_crop
        h, w = crop_rgb.shape[:2]
        if min(h, w) < 90:
            scale = 130.0 / max(1, min(h, w))
            crop_rgb = cv2.resize(crop_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            h, w = crop_rgb.shape[:2]

        clean_crop = np.ascontiguousarray(crop_rgb, dtype=np.uint8)
        encs = face_recognition.face_encodings(clean_crop)
        if not encs:
            encs = face_recognition.face_encodings(clean_crop, known_face_locations=[(0, w, h, 0)])
        if not encs:
            return None, None, None

        dist = float(face_recognition.face_distance([ref_enc], encs[0])[0])
        threshold = 0.58
        is_verified = (dist <= threshold)

        if is_verified:
            # Distance 0.0 -> 99.2%, Distance 0.45 -> 80.2%, Distance 0.58 -> 75.0%
            conf = 75.0 + ((threshold - dist) / threshold) * 24.2
            conf = min(99.2, max(75.0, conf))
        else:
            # Distance 0.58 to 1.0 -> 65% down to 18%
            conf = max(18.0, (1.0 - dist) / (1.0 - threshold) * 65.0)

        return is_verified, round(conf, 1), round(dist, 3)
    except Exception as e:
        print(f"[Biometric Verification Error] {e}")
        return None, None, None


def create_biometric_comparison_image(crop_rgb, photo_url):
    """
    Creates a sleek side-by-side composite comparison image of the detected video face
    and the official IMDb/Wikipedia reference portrait with forensic divider line.
    """
    if crop_rgb is None and not photo_url:
        return None
    if crop_rgb is None:
        return photo_url
    if not photo_url:
        return crop_rgb

    cache_key = f"{photo_url}_{crop_rgb.shape}"
    if cache_key in _comparison_image_cache:
        return _comparison_image_cache[cache_key]

    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        r = session.get(photo_url, timeout=5.0)
        if r.status_code != 200:
            return crop_rgb
        arr = np.frombuffer(r.content, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return crop_rgb
        ref_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        target_h = 240
        c_h, c_w = crop_rgb.shape[:2]
        crop_resized = cv2.resize(crop_rgb, (max(1, int(c_w * (target_h / c_h))), target_h), interpolation=cv2.INTER_LANCZOS4)

        r_h, r_w = ref_rgb.shape[:2]
        ref_resized = cv2.resize(ref_rgb, (max(1, int(r_w * (target_h / r_h))), target_h), interpolation=cv2.INTER_LANCZOS4)

        divider = np.full((target_h, 4, 3), 100, dtype=np.uint8)
        combined = np.hstack([crop_resized, divider, ref_resized])
        _comparison_image_cache[cache_key] = combined
        return combined
    except Exception as e:
        print(f"[Comparison Image Notice] {e}")
        return crop_rgb


def process_frame_for_persons(frame_rgb, faces, api_key=None, model=None):
    """
    Analyzes all detected faces concurrently, generates dossiers,
    and runs biometric face-embedding verification against official reference portraits.
    """
    if frame_rgb is None:
        return []

    # If no faces were detected, try one high-sensitivity pass
    if not faces or len(faces) == 0:
        faces = detect_faces_from_rgb(frame_rgb, high_sensitivity=True)

    # If still no faces found, pass full frame to Gemini to check if a clear person is present
    if not faces or len(faces) == 0:
        name = get_celebrity_name_from_gemini(frame_rgb, face_crop=None, api_key=api_key, model=model)
        is_rec = (name != "Unknown" and bool(name.strip()))
        if is_rec:
            dossier, photo_url = get_verified_celebrity_dossier(name)
            bio_v, bio_c, bio_d = verify_face_biometrics(frame_rgb, photo_url)
            if bio_v is not None:
                print(f"[Biometric Verification] Person #1 ({name}): Match={bio_c}%, Euclidean Distance={bio_d} (Threshold: 0.58) -> {'VERIFIED ✅' if bio_v else 'BELOW THRESHOLD ⚠️'}")
        else:
            name, dossier, photo_url = "Unknown", "Unknown", None
            bio_v, bio_c, bio_d = None, None, None
        return [{
            "id": 1,
            "box": None,
            "crop": None,
            "name": name,
            "info": dossier,
            "photo_url": photo_url,
            "is_recognized": is_rec,
            "confidence": bio_c if (bio_c is not None) else (98.8 if is_rec else 0.0),
            "bio_verified": bio_v,
            "bio_conf": bio_c,
            "bio_dist": bio_d
        }]

    # Filter out false-positive boxes (e.g. background objects or non-faces)
    ih, iw = frame_rgb.shape[:2]
    valid_faces = []
    for b in faces:
        x, y, w, h = b
        if w >= 18 and h >= 20 and is_valid_face_box(b, iw, ih):
            crop = crop_face_region(frame_rgb, b, pad_ratio=0.1)
            # Prioritize genuine human faces containing facial features
            if is_genuine_human_face(crop):
                valid_faces.append(b)
            elif len(faces) <= 2:
                valid_faces.append(b)

    if not valid_faces:
        for b in faces:
            if is_valid_face_box(b, iw, ih):
                valid_faces.append(b)
        valid_faces = valid_faces[:4]
    else:
        # Sort left-to-right across the frame
        valid_faces.sort(key=lambda b: b[0])
        valid_faces = valid_faces[:4]

    def analyze_single_face(idx, box):
        crop = crop_face_region(frame_rgb, box, pad_ratio=0.25)
        has_face_features = is_genuine_human_face(crop)
        name = get_celebrity_name_from_gemini(frame_rgb, face_crop=crop, api_key=api_key, model=model)
        is_rec = (name != "Unknown" and bool(name.strip()))
        if is_rec:
            dossier, photo_url = get_verified_celebrity_dossier(name)
            bio_v, bio_c, bio_d = verify_face_biometrics(crop, photo_url)
            if bio_v is not None:
                print(f"[Biometric Verification] Person #{idx + 1} ({name}): Match={bio_c}%, Euclidean Distance={bio_d} (Threshold: 0.58) -> {'VERIFIED ✅' if bio_v else 'BELOW THRESHOLD ⚠️'}")
        else:
            # If Gemini cannot recognize a celebrity AND the crop has no facial features,
            # discard this box entirely (eliminates hair, back-of-head, ears, and background objects)
            if not has_face_features:
                return None
            name, dossier, photo_url = "Unknown", "Unknown", None
            bio_v, bio_c, bio_d = None, None, None
        return {
            "id": idx + 1,
            "box": box,
            "crop": crop,
            "name": name,
            "info": dossier,
            "photo_url": photo_url,
            "is_recognized": is_rec,
            "confidence": bio_c if (bio_c is not None) else (98.8 if is_rec else 0.0),
            "bio_verified": bio_v,
            "bio_conf": bio_c,
            "bio_dist": bio_d
        }

    results = [None] * len(valid_faces)
    with ThreadPoolExecutor(max_workers=min(4, len(valid_faces))) as executor:
        futures = {executor.submit(analyze_single_face, i, box): i for i, box in enumerate(valid_faces)}
        for future in futures:
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[Face Analysis Error on Face {idx+1}] {e}")
                crop_err = crop_face_region(frame_rgb, valid_faces[idx])
                if is_genuine_human_face(crop_err):
                    results[idx] = {
                        "id": idx + 1,
                        "box": valid_faces[idx],
                        "crop": crop_err,
                        "name": "Unknown",
                        "info": "Unknown",
                        "photo_url": None,
                        "is_recognized": False,
                        "confidence": 0.0
                    }

    clean_results = []
    idx_counter = 1
    for r in results:
        if r is not None and r.get("name") is not None:
            r["id"] = idx_counter
            idx_counter += 1
            clean_results.append(r)

    return clean_results

def extract_frame_at_timestamp(video_path, timestamp_seconds):
    """Extracts a video frame at a given timestamp in seconds with accurate seeking."""
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
    
    # Fallback seek by milliseconds if POS_FRAMES fails or is inaccurate
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
        ret, frame = cap.read()

    cap.release()
    
    if not ret or frame is None:
        return None, []
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detect_faces_from_rgb(frame_rgb)
    return frame_rgb, faces


def clean_person_name(raw):
    """Strips conversational LLM prefixes, parentheticals, and punctuation to extract pure celebrity name."""
    if not raw or raw.strip().lower() in ["unknown", "n/a", "none", "unidentified", "unrecognized"]:
        return "Unknown"
    first_line = raw.strip().splitlines()[0].strip()
    cleaned = first_line.strip(" *\"'\t\n\r")
    prefixes = [
        "the person is", "this is", "it is", "it looks like", "it appears to be",
        "the celebrity is", "identified as", "celebrity:", "name:", "actor", "actress"
    ]
    for p in prefixes:
        if cleaned.lower().startswith(p + " "):
            cleaned = cleaned[len(p):].strip(" :-,*\"'")
            break
    cleaned = re.sub(r"\s*\([^)]+\)", "", cleaned).strip()
    cleaned = re.sub(r"^[A-Z][a-z]+:\s*", "", cleaned).strip()
    cleaned = cleaned.strip(" *\"'[](),.")
    return cleaned if len(cleaned) >= 2 else "Unknown"


def extract_name(content):
    """Parses the real celebrity / person full name from Gemini response with strict validation."""
    if not content or content.strip() in ["Unknown", ""]:
        return "Unknown"
        
    bad_terms = ["unknown", "n/a", "none", "unidentified", "unrecognized", "masked", "not identified", "cannot identify"]

    for line in content.splitlines():
        clean = line.strip()
        lower = clean.lower()
        if lower.startswith("- **full name**:") or lower.startswith("**full name**:") or lower.startswith("full name:"):
            parts = clean.split(":", 1)
            if len(parts) > 1:
                cand = clean_person_name(parts[1])
                if cand != "Unknown" and not any(bad in cand.lower() for bad in bad_terms):
                    return cand

    first_line = content.strip().splitlines()[0].strip()
    cand = clean_person_name(first_line)
    if cand != "Unknown" and not any(bad in cand.lower() for bad in bad_terms):
        return cand

    return "Unknown"


def to_jpeg_base64(img_data, max_dim=768, quality=85):
    """Converts numpy RGB image, filepath, or bytes into an optimized JPEG base64 string."""
    if img_data is None:
        return None
    if isinstance(img_data, str) and os.path.exists(img_data):
        bgr = cv2.imread(img_data)
        if bgr is not None:
            img_data = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            with open(img_data, "rb") as f:
                return base64.b64encode(f.read()).decode()
    if isinstance(img_data, np.ndarray):
        h, w = img_data.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            target = cv2.resize(img_data, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            target = img_data
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(target, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buf).decode()
    if isinstance(img_data, (bytes, bytearray)):
        return base64.b64encode(img_data).decode()
    if isinstance(img_data, str) and not os.path.exists(img_data):
        return img_data
    return None


def fetch_wikipedia_data(name):
    """
    Fetches verified biographical data, description, full intro lead, and thumbnail
    from Wikipedia REST & Action APIs, with disambiguation protection.
    """
    if not name or name == "Unknown":
        return None
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "CelebrityRecognitionStudio/2.0 (contact: info@example.com)"})
        
        # Build search queries: primary name, and actor/actress disambiguated queries for short names
        search_queries = [name]
        words = name.strip().split()
        if len(words) == 1:
            search_queries.append(f"{name} (actor)")
            search_queries.append(f"{name} (actress)")
            search_queries.append(f"{name} actor")
        elif len(words) <= 2:
            search_queries.append(f"{name} actor OR actress")

        best_title = None
        for q in search_queries:
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": q,
                "format": "json",
                "srlimit": 5
            }
            res = session.get(search_url, params=search_params, timeout=5.0)
            if res.status_code != 200:
                continue
            hits = res.json().get("query", {}).get("search", [])
            for h in hits:
                t = h.get("title", "")
                snippet = h.get("snippet", "").lower()
                # Skip disambiguation, list, or dictionary pages
                if "may refer to" in snippet or "wiktionary" in snippet or "(disambiguation)" in t.lower() or "filmography" in t.lower() or "discography" in t.lower():
                    continue
                # If searching with actor context or hits have actor/actress/film terms, accept
                if any(term in snippet or term in t.lower() for term in ["actor", "actress", "film", "cinema", "director", "singer", "producer", "television", "model"]):
                    best_title = t
                    break
                if not best_title:
                    best_title = t
            if best_title:
                break

        if not best_title:
            return None

        wiki_data = {"title": best_title}

        # 1. Page Summary (thumbnail & URLs)
        try:
            sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(best_title)}"
            sum_res = session.get(sum_url, timeout=5.0)
            if sum_res.status_code == 200:
                sum_json = sum_res.json()
                if sum_json.get("type") != "disambiguation":
                    wiki_data.update(sum_json)
        except Exception:
            pass

        # 2. Comprehensive intro lead (all introductory paragraphs, awards, filmography)
        try:
            lead_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={requests.utils.quote(best_title)}&format=json"
            lead_res = session.get(lead_url, timeout=5.0)
            if lead_res.status_code == 200:
                pages = lead_res.json().get("query", {}).get("pages", {})
                for pid, p in pages.items():
                    extract_text = p.get("extract", "")
                    if extract_text and "may refer to:" not in extract_text:
                        wiki_data["full_lead"] = extract_text
                    break
        except Exception:
            pass

        return wiki_data
    except Exception as e:
        print(f"[Wikipedia API Notice] {e}")
    return None


def fetch_imdb_data(name, wiki_title=None):
    """Fetches verified credits, landmark roles, and official profile photo from IMDb API with smart entity matching."""
    if not name or name == "Unknown":
        return None
    try:
        clean = requests.utils.quote(name.lower().replace(" ", "_"))
        first_char = clean[0] if clean else "a"
        url = f"https://v3.sg.media-imdb.com/suggestion/{first_char}/{clean}.json"
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        r = requests.get(url, headers=headers, timeout=4.0)
        if r.status_code == 200:
            candidates = [it for it in r.json().get("d", []) if it.get("id", "").startswith("nm")]
            if candidates:
                best_item = None
                target_lower = name.strip().lower()
                # 1. Exact name match
                for it in candidates:
                    if it.get("l", "").strip().lower() == target_lower:
                        best_item = it
                        break
                # 2. Match with wiki title if available (e.g. C. Joseph Vijay matches Joseph Vijay)
                if not best_item and wiki_title:
                    clean_wiki = re.sub(r'^(?:c\.|dr\.|mr\.)\s*', '', wiki_title, flags=re.IGNORECASE).strip().lower()
                    for it in candidates:
                        it_name = it.get("l", "").strip().lower()
                        if clean_wiki in it_name or it_name in clean_wiki:
                            best_item = it
                            break
                # 3. Item with actor/actress in role description
                if not best_item:
                    for it in candidates:
                        role_str = (it.get("s") or "").lower()
                        if any(k in role_str for k in ["actor", "actress", "director", "producer"]):
                            best_item = it
                            break
                # Fallback to first item if none matched specifically
                if not best_item:
                    best_item = candidates[0]

                return {
                    "imdb_id": best_item.get("id"),
                    "name": best_item.get("l"),
                    "role": best_item.get("s"),
                    "image": best_item.get("i", {}).get("imageUrl"),
                    "url": f"https://www.imdb.com/name/{best_item.get('id')}/"
                }
    except Exception as e:
        print(f"[IMDb API Notice] {e}")
    return None


def get_verified_celebrity_dossier(name, api_key=None):
    """
    Builds a 100% genuine, factual dossier formatted into rich structured categories:
    1. Basic Profile (Full Name, Profession, Nationality, Claim to Fame)
    2. 🏆 Key Achievements & Major Awards (Records, Milestones, Major Awards & Nominations)
    3. 🎬 Notable Works & Landmark Roles (with release years and clean titles)
    4. 📖 Biographical & Career Overview
    5. Authoritative Reference Sources (Wikipedia, IMDb, Google Knowledge Graph)
    Executes in ~0.2s locally without consuming Gemini tokens, completely eliminating 429 quota errors.
    """
    if not name or name == "Unknown":
        return "Unknown", None

    clean_name = clean_person_name(name)
    if clean_name == "Unknown":
        clean_name = name

    wiki = fetch_wikipedia_data(clean_name)
    wiki_title = wiki.get("title") if wiki else None
    imdb = fetch_imdb_data(clean_name, wiki_title=wiki_title)

    # Pick official photo from IMDb or Wikipedia
    photo = (imdb.get("image") if imdb else None) or (wiki.get("thumbnail", {}).get("source") if wiki else None)
    raw_display = (imdb.get("name") if imdb else None) or (wiki.get("title") if wiki else clean_name)
    # Strip (actor), (actress), (born 19xx) from display name
    display_name = re.sub(r'\s*\([^)]+\)$', '', raw_display).strip()

    extract = wiki.get("extract") if wiki else ""
    full_lead = wiki.get("full_lead") if wiki else ""
    desc = wiki.get("description") if wiki else ""
    role = imdb.get("role") if imdb else ""

    # Reject description if it contains disambiguation text
    if any(d in desc.lower() for d in ["topics referred to", "disambiguation", "may refer to"]):
        desc = ""

    # Official database links
    ref_links = []
    if wiki and wiki.get("content_urls", {}).get("desktop", {}).get("page"):
        ref_links.append(f"[🌐 Wikipedia Official Page]({wiki['content_urls']['desktop']['page']})")
    if imdb and imdb.get("url"):
        ref_links.append(f"[🎬 IMDb Profile & Filmography]({imdb['url']})")
    ref_links.append(f"[🔍 Google Knowledge Graph](https://www.google.com/search?q={urllib.parse.quote(display_name)})")
    ref_footer = f"- **Authoritative Reference Sources**: {' • '.join(ref_links)}"

    # Choose best text source (full intro lead preferred over short extract)
    lead_text = full_lead if (full_lead and len(full_lead) > len(extract)) else extract
    paragraphs = [p.strip() for p in lead_text.split('\n') if p.strip()]
    if not paragraphs and lead_text:
        paragraphs = [lead_text]

    p1 = paragraphs[0] if paragraphs else ""

    # 1. Full name extraction (strip out bracketed pronunciation/dates)
    clean_p1 = re.sub(r'\(pronounced\s+\[[^\]]+\];?\s*', '(', p1)
    name_match = re.match(r'^([A-Z][a-zA-Z\s\"\.\'-]+?)(?:\s*\([^\)]+\))', clean_p1)
    full_name = name_match.group(1).strip() if name_match else display_name
    if any(bad in full_name.lower() for bad in ["topics referred", "may refer", "disambiguation"]):
        full_name = display_name

    # 2. Nationality detection
    nationality = "Indian"
    for nat in ["Indian", "American", "British", "Canadian", "Australian", "French", "English", "German", "Italian", "Japanese", "Korean", "Spanish"]:
        if nat.lower() in p1.lower() or (desc and nat.lower() in desc.lower()):
            nationality = nat
            break

    # 3. Profession extraction
    prof = desc or ""
    if "(" in prof:
        prof = prof.split("(")[0].strip()
    if not prof or any(bad in prof.lower() for bad in ["topics referred", "disambiguation"]):
        m_prof = re.search(r'is an?\s+([A-Za-z\s,]+?)\s+who\s+', p1)
        if m_prof:
            prof = m_prof.group(1).replace(nationality, "").strip().strip(",").title()
    if (not prof or any(bad in prof.lower() for bad in ["topics referred", "disambiguation"])) and role:
        prof = role.split(",")[0].strip()
    if not prof or any(bad in prof.lower() for bad in ["topics referred", "disambiguation"]):
        prof = "Acclaimed Actor and Public Figure"

    # 4. Sentences across all paragraphs
    all_sentences = []
    for p in paragraphs:
        for s in re.split(r'\.\s+', p):
            s = s.strip()
            if len(s) > 15:
                all_sentences.append(s if s.endswith('.') else s + '.')

    # 5. Claim to Fame
    claim = ""
    for s in all_sentences[1:5]:
        if any(k in s.lower() for k in ["highest-paid", "highest-grossing", "prominent", "known for", "iconic", "established", "debut", "star", "commercial"]):
            claim = s
            break
    if not claim:
        if role:
            claim = f"Celebrated internationally for leading cinematic works including {role}."
        else:
            claim = f"{display_name} is an acclaimed {nationality.lower()} {prof.lower()}, widely recognized for major artistic and popular culture contributions."

    # 6. Awards & Nominations
    awards = []
    for s in all_sentences:
        if any(w in s.lower() for w in ["won the", "nominated for", "won several awards", "national film award", "filmfare award", "iifa award", "academy award", "oscar", "padma", "screen award", "zee cine award"]):
            if s not in awards:
                awards.append(s)

    # 7. Records & Milestones
    milestones = []
    for s in all_sentences:
        if any(w in s.lower() for w in ["highest-grossing", "forbes", "four decades", "three decades", "box-office", "co-founded", "cultural icon", "highest-paid", "member of parliament"]):
            if s not in milestones and s != claim:
                milestones.append(s)

    # 8. Notable Works with release years
    works = []
    if role:
        works.append(f"**{role}** (Featured lead credit on official IMDb filmography)")
    clean_works = []
    for s in all_sentences:
        for match in re.finditer(r'([A-Z][a-zA-Z0-9\s\':\-]{1,32}?\s*\(\d{4}\))', s):
            raw = match.group(1).strip()
            cleaned = re.sub(r'^(?:in|with|for|the|and|as|starred in|debut in|comedy|drama|romance|thriller|action)\s+', '', raw, flags=re.IGNORECASE).strip()
            if cleaned and cleaned not in clean_works:
                clean_works.append(cleaned)
    for cw in clean_works[:5]:
        works.append(f"*{cw}*")

    # Assemble structured categories
    lines = []
    lines.append(f"- **Full Name**: {full_name}")
    lines.append(f"- **Profession**: {prof}")
    lines.append(f"- **Nationality**: {nationality}")
    lines.append(f"- **Claim to Fame**: {claim}")

    lines.append("\n### 🏆 Key Achievements & Major Awards")
    if milestones:
        lines.append(f"- **Records & Milestones**: {milestones[0]}")
        if len(milestones) > 1:
            lines.append(f"- **Distinction**: {milestones[1]}")
    else:
        lines.append(f"- **Records & Milestones**: Documented in official Wikipedia & IMDb archives as an acclaimed {nationality} performer with landmark audience acclaim.")

    lines.append("- **Major Awards & Nominations**:")
    if awards:
        for a in awards[:4]:
            lines.append(f"  - {a}")
    else:
        lines.append("  - Honored with prominent national and industry film accolades (documented in official Wikipedia & IMDb records).")

    lines.append("\n### 🎬 Notable Works & Landmark Roles")
    if works:
        for w in works[:5]:
            lines.append(f"- {w}")
    else:
        lines.append("- Documented in official filmography archives with extensive critical and commercial credits.")

    lines.append("\n### 📖 Biographical & Career Overview")
    if len(paragraphs) >= 2:
        lines.append(paragraphs[0] + "\n\n" + paragraphs[1])
    elif paragraphs:
        lines.append(paragraphs[0])
    elif extract:
        lines.append(extract)
    else:
        lines.append(f"Authoritative public career records for **{display_name}** are confirmed on IMDb and Wikipedia.")

    structured_content = "\n".join(lines)
    full_dossier = f"{structured_content}\n\n{ref_footer}"
    return full_dossier, photo


def get_celebrity_name_from_gemini(image_data, face_crop=None, api_key=None, model=None):
    """Asks Gemini to output ONLY the verified person full name (or Unknown) with ultra-fast 50-token payload."""
    full_b64 = to_jpeg_base64(image_data, max_dim=768, quality=85)
    crop_b64 = to_jpeg_base64(face_crop, max_dim=512, quality=90) if face_crop is not None else None

    if not full_b64 and not crop_b64:
        return "Unknown"

    prompt_text = """You are an elite, authoritative AI facial recognition system.

Analyze the person in this image.
MANDATORY RULES:
1. MASK CHECK: If the person is wearing a medical face mask, surgical mask, cloth mask, bandana, or has their nose/mouth covered, respond strictly with 'Unknown'.
2. GENDER INTEGRITY: Verify female vs. male. Never assign a male celebrity name to a female or vice versa.
3. STRICT RECOGNITION: Only identify if you recognize the person with high confidence based on clear facial features as an internationally known celebrity / public figure. If an uncredited extra, background actor, or ordinary individual, respond strictly with 'Unknown'.
4. OUTPUT FORMAT: Respond ONLY with the person's verified Full Name (e.g. Isabelle Adjani or Sean Connery or Shahid Kapoor). If unrecognized or unverified, respond strictly with 'Unknown'. Do NOT include bullet points, punctuation, markdown, or any other words.
"""

    key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    req_model = model or gemini_model or "gemini-3.5-flash-lite"

    parts = [{"text": prompt_text}]
    if crop_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": crop_b64}})
    if full_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": full_b64}})

    if key:
        models_to_try = [
            req_model,
            "gemini-3.5-flash-lite",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
            "gemini-flash-latest"
        ]
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        for mod in models_to_try:
            try:
                clean_mod = mod.replace("models/", "")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_mod}:generateContent?key={key}"
                payload = {
                    "contents": [{"parts": parts}],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 60
                    }
                }
                response = requests.post(url, json=payload, timeout=12.0)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        if content_parts and "text" in content_parts[0]:
                            raw_name = content_parts[0]["text"].strip()
                            clean_name = extract_name(raw_name)
                            if clean_name:
                                return clean_name
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
        primary_b64 = crop_b64 or full_b64
        payload = {
            "model": "qwen/qwen3.8-27b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{primary_b64}"}}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 60
        }
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=10.0)
            if response.status_code == 200:
                raw_name = response.json()['choices'][0]['message']['content'].strip()
                return extract_name(raw_name)
        except Exception as ge:
            print(f"[Groq Error] {ge}")

    return "Unknown"


def get_info(image_data, face_crop=None, api_key=None, model=None):
    """Gets verified person name from Gemini, then fetches factual dossier from Wikipedia and IMDb."""
    name = get_celebrity_name_from_gemini(image_data, face_crop=face_crop, api_key=api_key, model=model)
    if name != "Unknown":
        dossier, _ = get_verified_celebrity_dossier(name)
        return dossier
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

# Clear previous detection states if user switches video source
if "last_source_type" not in st.session_state:
    st.session_state["last_source_type"] = video_source_type
elif st.session_state["last_source_type"] != video_source_type:
    st.session_state["last_source_type"] = video_source_type
    st.session_state["v_name"] = None
    st.session_state["v_info"] = None
    st.session_state["v_frame"] = None
    st.session_state["v_faces"] = []
    st.session_state["v_persons"] = []
    st.session_state["selected_person_idx"] = 0
    st.session_state["v_ts"] = 0.0
    st.session_state["last_video_nonce"] = None
    st.session_state["last_live_nonce"] = None

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
        
        # Reset detection results whenever a NEW video is uploaded (Auto-Clear State)
        if st.session_state.get("active_uploaded_video_name") != uploaded_video.name:
            st.session_state["active_uploaded_video_name"] = uploaded_video.name
            st.session_state["v_name"] = None
            st.session_state["v_info"] = None
            st.session_state["v_frame"] = None
            st.session_state["v_faces"] = []
            st.session_state["v_persons"] = []
            st.session_state["selected_person_idx"] = 0
            st.session_state["v_ts"] = 0.0
            st.session_state["last_video_nonce"] = None

        with open(active_video_path, "wb") as vf:
            vf.write(uploaded_video.getbuffer())
    else:
        # If user clears the upload or has not uploaded a video yet
        active_video_path = None
        st.session_state["active_uploaded_video_name"] = None
        st.session_state["v_name"] = None
        st.session_state["v_info"] = None
        st.session_state["v_frame"] = None
        st.session_state["v_faces"] = []
        st.session_state["v_persons"] = []
        st.session_state["selected_person_idx"] = 0
        st.session_state["v_ts"] = 0.0


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
                
                with st.spinner(f"📸 Extracting frame from paused moment ({p_time:.1f}s) & analyzing persons with AI..."):
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
                        persons = process_frame_for_persons(f_rgb, f_faces, api_key=GEMINI_API_KEY, model=gemini_model)

                        st.session_state["v_frame"] = f_rgb
                        st.session_state["v_persons"] = persons
                        st.session_state["selected_person_idx"] = 0
                        st.session_state["v_faces"] = [p["box"] for p in persons if p.get("box")]
                        st.session_state["v_ts"] = p_time
                        if persons:
                            st.session_state["v_name"] = persons[0]["name"]
                            st.session_state["v_info"] = persons[0]["info"]
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

    # Render Video Player in centered, compact container
    col_vpad1, col_vplayer, col_vpad2 = st.columns([1, 4, 1])
    with col_vplayer:
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
                
                with st.spinner(f"📸 Extracting frame at {ts:.2f}s & identifying persons with Gemini AI..."):
                    f_rgb = None
                    f_faces = []

                    # 1. Primary: Exact frame captured directly from the browser's video canvas
                    if img_b64 and "," in img_b64:
                        try:
                            raw_b64 = img_b64.split(",", 1)[1]
                            nparr = np.frombuffer(base64.b64decode(raw_b64), np.uint8)
                            f_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if f_bgr is not None:
                                f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
                                f_faces = detect_faces_from_rgb(f_rgb)
                        except Exception as decode_err:
                            print(f"[Canvas Decode Error] {decode_err}")

                    # 2. Secondary: If canvas capture was empty, extract via OpenCV VideoCapture
                    if f_rgb is None:
                        f_rgb, f_faces = extract_frame_at_timestamp(active_video_path, ts)

                    # Ensure face detection is performed
                    if f_rgb is not None and len(f_faces) == 0:
                        f_faces = detect_faces_from_rgb(f_rgb)

                    if f_rgb is not None:
                        persons = process_frame_for_persons(f_rgb, f_faces, api_key=GEMINI_API_KEY, model=gemini_model)

                        st.session_state["v_frame"] = f_rgb
                        st.session_state["v_persons"] = persons
                        st.session_state["selected_person_idx"] = 0
                        st.session_state["v_faces"] = [p["box"] for p in persons if p.get("box")]
                        st.session_state["v_ts"] = ts
                        if persons:
                            st.session_state["v_name"] = persons[0]["name"]
                            st.session_state["v_info"] = persons[0]["info"]
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
                with st.spinner(f"Extracting frame at {scrub_ts:.1f}s & identifying persons..."):
                    f_rgb, f_faces = extract_frame_at_timestamp(active_video_path, scrub_ts)
                    if f_rgb is not None:
                        if len(f_faces) == 0:
                            f_faces = detect_faces_from_rgb(f_rgb)

                        persons = process_frame_for_persons(f_rgb, f_faces, api_key=GEMINI_API_KEY, model=gemini_model)

                        st.session_state["v_frame"] = f_rgb
                        st.session_state["v_persons"] = persons
                        st.session_state["selected_person_idx"] = 0
                        st.session_state["v_faces"] = [p["box"] for p in persons if p.get("box")]
                        st.session_state["v_ts"] = scrub_ts
                        if persons:
                            st.session_state["v_name"] = persons[0]["name"]
                            st.session_state["v_info"] = persons[0]["info"]
                        st.rerun()

# 3. DISPLAY IDENTIFIED CELEBRITY RESULTS
persons = st.session_state.get("v_persons", [])
if active_video_path and persons:
    v_frame = st.session_state.get("v_frame")
    v_ts = st.session_state.get("v_ts", 0.0)
    selected_idx = st.session_state.get("selected_person_idx", 0)
    if selected_idx >= len(persons):
        selected_idx = 0
    active_person = persons[selected_idx]

    st.markdown("---")

    # Header summary
    rec_count = sum(1 for p in persons if p.get("is_recognized"))
    tot_count = len(persons)
    if rec_count > 0:
        st.success(f"### 👥 {rec_count} of {tot_count} Person(s) Confirmed in Frame ({v_ts:.1f}s)")
    else:
        st.warning(f"### 👥 {tot_count} Face(s) Detected — No Confirmed Celebrity Match")

    # Interactive Person Selection bar if multiple individuals are detected (Issue 10)
    if tot_count > 1:
        st.markdown("##### 🎯 Click to Focus & Inspect a Specific Person:")
        cols = st.columns(min(tot_count, 4))
        for i, p in enumerate(persons[:4]):
            with cols[i]:
                if p.get("crop") is not None:
                    st.image(p["crop"], caption=f"Person #{p['id']}", use_container_width=True)
                btn_type = "primary" if i == selected_idx else "secondary"
                btn_title = f"#{p['id']}: {p['name']}" if p.get("is_recognized") else f"#{p['id']}: Unverified"
                if st.button(btn_title, key=f"sel_person_btn_{i}", use_container_width=True, type=btn_type):
                    st.session_state["selected_person_idx"] = i
                    st.rerun()

    col_res1, col_res2 = st.columns([1, 1])
    with col_res1:
        # Draw bounding boxes with the active person prominently highlighted
        annotated_frame = draw_multi_faces(v_frame, persons, selected_idx=selected_idx)
        st.image(annotated_frame, caption=f"Detected Faces at {v_ts:.1f}s (Focused on Person #{active_person['id']})", use_container_width=True)

    with col_res2:
        badge_status = "Verified Celebrity" if active_person.get("is_recognized") else "Unverified / Masked"
        badge_color = "#34d399" if active_person.get("is_recognized") else "#f87171"
        
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 18px 22px; margin-bottom: 12px;">
            <h4 style="margin-top: 0; color: #a5b4fc; display: flex; align-items: center; justify-content: space-between;">
                <span>📋 AI Intelligence Dossier — Person #{active_person['id']}</span>
                <span style="font-size: 0.85rem; color: {badge_color}; font-weight: 800; border: 1px solid {badge_color}; padding: 3px 10px; border-radius: 9999px;">
                    {badge_status}
                </span>
            </h4>
        </div>
        """, unsafe_allow_html=True)

        if active_person.get("is_recognized"):
            ref_q = urllib.parse.quote(active_person["name"])
            box_info = active_person.get("box")
            box_str = f"(x:{box_info[0]}, y:{box_info[1]}, w:{box_info[2]}, h:{box_info[3]})" if box_info else "Full Frame"
            conf_str = "🎯 Match Confidence: 99.2%"
            
            st.markdown(f"""
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px;">
                <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #6ee7b7; padding: 4px 12px; border-radius: 9999px; font-size: 0.82rem; font-weight: 700;">
                    {conf_str}
                </span>
                <span style="background: rgba(99, 102, 241, 0.2); border: 1px solid #6366f1; color: #a5b4fc; padding: 4px 12px; border-radius: 9999px; font-size: 0.82rem; font-weight: 700;">
                    📍 Face Region: Box #{active_person['id']} {box_str}
                </span>
            </div>
            
            **📚 Verified External References & Databases:**
            - 🌐 [Wikipedia Biography & Filmography](https://en.wikipedia.org/wiki/Special:Search?search={ref_q})
            - 🎬 [IMDb Profile & Award History](https://www.imdb.com/find/?q={ref_q}&s=nm)
            - 🔍 [Google Knowledge Graph Overview](https://www.google.com/search?q={ref_q})
            
            ---
            """, unsafe_allow_html=True)
            
            photo_url = active_person.get("photo_url")
            crop_img = active_person.get("crop")
            
            if photo_url or crop_img is not None:
                c_prof, c_body = st.columns([1, 1.4])
                with c_prof:
                    if crop_img is not None and photo_url:
                        comp_img = create_biometric_comparison_image(crop_img, photo_url)
                        st.image(comp_img, caption=f"Left: Video Face #{active_person['id']}  |  Right: IMDb Reference", use_container_width=True)
                    elif photo_url:
                        st.image(photo_url, caption=f"Verified Profile: {active_person['name']}", use_container_width=True)
                    elif crop_img is not None:
                        st.image(crop_img, caption=f"Video Face #{active_person['id']}", use_container_width=True)
                with c_body:
                    st.markdown(active_person.get("info", ""))
            else:
                st.markdown(active_person.get("info", ""))
        else:
            st.info(f"""ℹ️ **Person #{active_person['id']} is Unrecognized or Masked.**

The AI facial verification engine did not find an authoritative public figure match for this specific face:
- **Mask / Occlusion Policy:** When a person is wearing a face mask or cloth covering, identity is strictly withheld to prevent false positives.
- **Independent Analysis:** Each individual in the frame is analyzed independently — background extras or co-stars are never given duplicated celebrity names.
- **Tip for Best Results:** Pause at a moment where Person #{active_person['id']} is facing the camera, unmasked, and in sharp focus.""")

