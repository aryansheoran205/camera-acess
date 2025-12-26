from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2

app = FastAPI()

# Open camera
camera = cv2.VideoCapture(0)

# Load Haar Cascade safely
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("Cascade loaded:", not detector.empty())  # thiiiiss is for kam kara sa ke koni
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

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

        # Encode frame
        ret, buffer = cv2.imencode(".jpg", frame)
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
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
