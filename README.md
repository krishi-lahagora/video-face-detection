# 🌟 Celebrity Video & Live TV Recognition AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Face%20Detection-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

An AI-powered video and live TV application that identifies celebrities, cinema actors, and public figures directly from **video streams**, **HLS live TV (.m3u8)**, and **uploaded video files** using **Google Gemini Vision AI** and **OpenCV Face Detection**.

</div>

---

## ⚡ Highlights & Features

- 📡 **Live TV & HLS Streaming**: Play live TV broadcast streams (`.m3u8`) with full video playback and audio.
- 🎯 **One-Click Pause-Time Detection**: Pause the video at any moment when a celebrity appears, and capture that exact frame from the screen to detect who is on screen.
- 🤖 **Google Gemini Vision Intelligence**: Powered by `gemini-3.8-flash` (with automated fallback to `gemini-2.5-flash` and `gemini-1.5-flash`) for recognition and biographical metadata.
- 👤 **Face Bounding Box & Annotation**: OpenCV detects the exact facial coordinates and labels the actor's real name directly onto the image.
- 🎬 **Video Upload Support**: Upload any movie or video file (`.mp4`, `.mov`, `.avi`, `.mkv`).
- ⏱️ **Timestamp Scrubber**: Jump to any timestamp in the video to analyze specific frames.

---

## 📁 Project Structure

```
├── app.py                             # Main Streamlit application
├── components/
│   ├── live_tv_player/                # Custom HLS.js live TV player with canvas frame capture
│   └── pause_detector/                # HTML component for pause-time video frame extraction
├── haarcascade_frontalface_default.xml # OpenCV Haar Cascade model
├── requirements.txt                   # Python dependencies
├── .env.example                       # Example environment configuration
└── static/
    └── uploads/                       # Temporary upload folder
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/DataScientist00/Celebrity-Face-Detection-GenerativeAI.git
cd Celebrity-Face-Detection-GenerativeAI
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Add your free **Google Gemini API Key** in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```
*(You can get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/)).*

You can also paste the API key directly into the sidebar in the UI while running the app.

### 5. Run the application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** (or the port shown in your terminal) in your browser.

---

## 🎯 How to Use

1. **Live TV Stream**:
   - Choose **`📡 Live TV Stream (M3U8)`**.
   - Watch the live stream in the embedded player.
   - Pause the stream when a celebrity appears and click **`🎯 Detect Person from This Paused Frame`**.
2. **Video File Upload**:
   - Choose **`📤 Upload Video / Movie File`**.
   - Upload any video, scrub to the desired scene, and click the glowing detect button.
3. **View AI Results**:
   - The app displays the marked face with the actor's real name.
   - Full dossier: **Full Name**, **Profession**, **Nationality**, and **Famous Works / Movies**.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Computer Vision**: [OpenCV](https://opencv.org/)
- **Vision AI**: [Google Gemini Flash (v1beta REST)](https://ai.google.dev/)
- **Video Streaming**: [Hls.js](https://github.com/video-dev/hls.js)
