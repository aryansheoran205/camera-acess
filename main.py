from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2

app = FastAPI()

camera = cv2.VideoCapture(0)

# load detector
detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

@app.get("/", response_class=HTMLResponse)
def home():
    return open("index.html").read()

@app.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
