from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import time
import os

app = FastAPI()

# Open camera with V4L2 backend
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

# --- THE FIX FOR TIMEOUTS ---
# 1. Force MJPEG format (Compresses data so it doesn't time out the USB bridge)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# 2. Lower the resolution to reduce bandwidth stress on the WSL bridge
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# ----------------------------

# Load Haar Cascade safely
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("Cascade loaded:", not detector.empty())  # thiiiss is for kam kara sa ke koni

# Global variable to hold the latest frame for the capture button
last_frame = None

# Ensure the folder exists
os.makedirs("captures", exist_ok=True)

def gen_frames():
    global last_frame
    while True:
        success, frame = camera.read()
        if not success:
            # Instead of breaking, we wait and retry to bypass temporary timeouts
            time.sleep(0.1)
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Face detection
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save a copy for the capture endpoint
        last_frame = frame.copy()

        # Encode frame
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # MJPEG stream
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.get("/", response_class=HTMLResponse)
def home():
    try:
        return open("index.html").read()
    except FileNotFoundError:
        return "<h1>index.html not found!</h1>"

@app.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/capture")
def capture():
    global last_frame
    if last_frame is not None:
        filename = f"captures/face_{int(time.time())}.jpg"
        cv2.imwrite(filename, last_frame)
        return {"status": "success", "filename": filename}
    return {"status": "error", "message": "No frame available"}

@app.on_event("shutdown")
def shutdown_event():
    camera.release()
