from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import os
import time

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# video capture is a class
camera = cv2.VideoCapture(0)

# detector is object and class is cassavuybhahu
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Cascade loaded:", not detector.empty())  # thiiiss is for kam kara sa ke koni

last_frame = None

os.makedirs("captures", exist_ok=True)

# yha se function start
def gen_frames():
    global last_frame
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # function
        gray = cv2.equalizeHist(gray)

        # Face detection
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,      # ye method ha ek jo detection ma kam ava sa
            minNeighbors=4,
            minSize=(60, 60)
        )

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # yo vo area banava go jo face ka uper ava go

        last_frame = frame.copy()

        # Encode frame
        ret, buffer = cv2.imencode(".jpg", frame)  # yo jpeg format ma change kara go
        frame = buffer.tobytes()

        # MJPEG stream
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.get("/", response_class=HTMLResponse)
def home():
    return open("index.html").read()


@app.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(),  # ya ek class sa jo grt_frame ka andar ha
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/capture")
def capture_image():
    if last_frame is None:
        return {"error": "No frame available"}

    filename = f"captures/capture_{int(time.time())}.jpg"
    cv2.imwrite(filename, last_frame)

    return {"filename": filename}
