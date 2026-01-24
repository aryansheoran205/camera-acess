from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import os
import threading
import requests 

# --- APP INITIALIZATION ---
# 'app' ek object hai jo FastAPI class se bana hai. Yeh hamara web server handle karta hai.
app = FastAPI()

# --- TELEGRAM CONFIG ---
# Yeh variables bot ko batate hain ki message kahan aur kaise bhejna hai.
TELEGRAM_TOKEN = "8597784933:AAF8xO734vqhimwsbDgVjIPiQlFN-wmTlg4"
TELEGRAM_CHAT_ID = "2069062436"

# --- CAMERA SETUP ---
# 'camera' variable mein hum hardware camera ka access store karte hain.
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- MODELS LOAD KARNA ---
# 'CascadeClassifier' ek class hai jo pehle se trained files (.xml) ko load karti hai.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
helmet_cascade = cv2.CascadeClassifier("helmet_cascade.xml") 

os.makedirs("captures", exist_ok=True)

# --- GLOBAL VARIABLES ---
# 'latest_raw_frame' ek variable hai jo sabse naya camera image hold karta hai.
latest_raw_frame = None  
# 'last_notification_time' float value store karta hai (time in seconds).
last_notification_time = 0 
# 'lock' threading class ka object hai jo data ko crash hone se bachata hai.
lock = threading.Lock() 

def capture_thread():
    """Yeh ek Function hai jo background mein chalta hai."""
    global latest_raw_frame
    while True:
        success, frame = camera.read()
        if success:
            with lock:
                latest_raw_frame = frame

# Thread function ko start kar raha hai.
threading.Thread(target=capture_thread, daemon=True).start()

def send_telegram_alert(photo_path):
    """Yeh Function Telegram par photo upload karne ka kaam karta hai."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo:
            payload = {'chat_id': TELEGRAM_CHAT_ID, 'caption': "⚠️ ALERT: Bina Helmet ke rider dikha!"}
            files = {'photo': photo}
            requests.post(url, data=payload, files=files)
    except Exception as e:
        print(f"Telegram Error: {e}")

def process_frame(frame):
    """Yeh sabse bada Function hai jo image par math apply karke shapes banata hai."""
    global last_notification_time
    
    # 'gray' ek variable hai jo BGR image ko Black & White mein convert karke store karta hai.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- 1. HELMET DETECTION ---
    # 'helmets' ek list (array) hai jisme detect kiye gaye objects ke coordinates hote hain.
    helmets = helmet_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # 'helmet_detected' ek Boolean variable hai (True ya False).
    helmet_detected = False
    for (x, y, w, h) in helmets:
        helmet_detected = True
        # cv2.rectangle ek inbuilt function hai jo rectangle banata hai.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, "HELMET", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- 2. FACE DETECTION ---
    if not helmet_detected:
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "NO HELMET!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Logic: Agar time ka gap 15s se zyada hai toh alert bhejo.
            if time.time() - last_notification_time > 15:
                filename = f"captures/alert_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame) 
                send_telegram_alert(filename) 
                last_notification_time = time.time()

    # --- 3. HAND DETECTION ---
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # 'lower' aur 'upper' variables array format mein color limits store karte hain.
    lower = np.array([0, 30, 60], dtype="uint8")
    upper = np.array([20, 150, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 'contours' variable hand ki boundary lines ki list hold karta hai.
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_cnt = max(contours, key=cv2.contourArea) 
        if cv2.contourArea(max_cnt) > 3000:
            x, y, w, h = cv2.boundingRect(max_cnt)
            cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), (255, 0, 255), 2)
            
            # 'hull' hand ki outer boundary ko smooth karne wala variable hai.
            hull = cv2.convexHull(max_cnt)
            cv2.drawContours(frame, [hull], -1, (255, 255, 255), 2)
            
            hull_idx = cv2.convexHull(max_cnt, returnPoints=False)
            defects = cv2.convexityDefects(max_cnt, hull_idx)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_cnt[s][0])
                    if d > 10000: 
                        cv2.circle(frame, start, 6, [0, 0, 255], -1)

    return frame

def gen_frames():
    """Yeh Function video data ko generator ki tarah stream karta hai."""
    global latest_raw_frame
    while True:
        with lock:
            if latest_raw_frame is None:
                continue
            frame = latest_raw_frame.copy()

        processed = process_frame(frame) 
        ret, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- API ROUTES (GET METHODS) ---
@app.get("/", response_class=HTMLResponse)
def index():
    return open("index.html").read()

@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("shutdown")
def shutdown():
    camera.release()
