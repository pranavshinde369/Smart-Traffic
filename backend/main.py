"""
SMC-Niyantran Backend
Solapur Mobility Control & Niyantran
FastAPI + YOLOv8 + OpenCV Traffic Intelligence Engine
"""

import cv2
import time
import threading
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ─────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────
app = FastAPI(
    title="SMC-Niyantran API",
    description="Solapur Mobility Control – AI Traffic Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Constants & Configuration
# ─────────────────────────────────────────────

# YOLO vehicle class IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

# Virtual Geofence / No-Parking Zone (x1, y1, x2, y2) in pixels
ZONE = (50, 200, 350, 450)

# Path to dummy traffic video (must exist in same directory as main.py)
VIDEO_PATH = Path(__file__).parent / "traffic_feed.mp4"

# YOLO model – downloads yolov8n.pt automatically on first run
MODEL_PATH = "yolov8n.pt"

# Carbon emission constants
# Average car idles at ~170g CO2/min; reduced idling from smarter signals
CARBON_PER_VEHICLE_PER_SECOND = 0.0028  # kg CO2 saved per vehicle per second of green time reduction

# ─────────────────────────────────────────────
# Shared State (thread-safe via GIL for simple types)
# ─────────────────────────────────────────────
state = {
    "vehicle_count": 0,
    "encroachment_alert": False,
    "dynamic_green_time": 20,
    "carbon_saved_kg": 0.0,
    "frame": None,          # Latest JPEG-encoded processed frame
    "running": False,
}

state_lock = threading.Lock()

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def is_in_zone(cx: int, cy: int, zone: tuple) -> bool:
    """Check if a center point (cx, cy) is inside the rectangular zone."""
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2


def compute_green_time(count: int) -> int:
    """
    RL-Optimizer: Dynamic green signal duration based on vehicle density.
    Formula: clamp(count * 3, 20, 120) seconds
    """
    return min(120, max(20, count * 3))


def compute_carbon_saved(count: int, green_time: int) -> float:
    """
    Eco-Niyantran: Estimate CO2 saved vs. fixed 60s signal.
    If our dynamic green time < 60s, vehicles idle less → carbon saved.
    Positive when dynamic_green_time < 60 (less idling than baseline).
    """
    baseline_green = 60  # seconds (fixed legacy signal)
    time_saved = max(0, baseline_green - green_time)
    return round(count * time_saved * CARBON_PER_VEHICLE_PER_SECOND, 3)


def draw_zone(frame: np.ndarray, zone: tuple) -> np.ndarray:
    """Draw the geofenced no-parking zone on frame."""
    x1, y1, x2, y2 = zone
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 100, 255), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
    cv2.putText(
        frame, "NO PARKING ZONE",
        (x1 + 4, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 200, 255), 1, cv2.LINE_AA
    )
    return frame


def draw_hud(frame: np.ndarray, count: int, green_time: int, carbon: float, alert: bool) -> np.ndarray:
    """Overlay HUD stats onto the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "SMC-NIYANTRAN | MARKET YARD JN.",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Vehicles: {count}  |  Green: {green_time}s  |  CO2 Saved: {carbon}kg",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if alert:
        # Flashing red alert bar at bottom
        tick = int(time.time() * 2) % 2
        if tick == 0:
            cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 200), -1)
            cv2.putText(frame, "!! ENCROACHMENT ALERT – E-CHALLAN DRAFTED !!",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
# Background Video Processing Thread
# ─────────────────────────────────────────────

def video_processing_loop():
    """
    Runs in a daemon thread. Loops the traffic_feed.mp4 continuously,
    runs YOLOv8 inference on each frame, updates shared state.
    Falls back to a synthetic demo feed if video file is not found.
    """
    model = YOLO(MODEL_PATH)

    # Attempt to open real video; fall back to synthetic frames
    use_synthetic = not VIDEO_PATH.exists()
    cap = None

    if not use_synthetic:
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        if not cap.isOpened():
            use_synthetic = True

    with state_lock:
        state["running"] = True

    accumulated_carbon = 0.0
    frame_idx = 0

    while True:
        # ── Get Frame ────────────────────────────────
        if use_synthetic:
            frame = generate_synthetic_frame(frame_idx)
            frame_idx += 1
            time.sleep(0.05)  # ~20 FPS synthetic
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            frame = cv2.resize(frame, (640, 480))

        # ── YOLOv8 Inference ─────────────────────────
        results = model(frame, classes=VEHICLE_CLASSES, conf=0.35, verbose=False)

        vehicle_count = 0
        encroachment = False

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                in_zone = is_in_zone(cx, cy, ZONE)

                if in_zone:
                    encroachment = True
                    color = (0, 0, 255)      # RED – encroachment
                    label_color = (0, 0, 255)
                else:
                    color = (0, 255, 100)    # GREEN – normal
                    label_color = (0, 255, 100)

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Center dot
                cv2.circle(frame, (cx, cy), 4, color, -1)

                # Label
                label = f"{'CAR' if cls_id==2 else 'BIKE' if cls_id==3 else 'BUS' if cls_id==5 else 'TRUCK'} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1, cv2.LINE_AA)

                vehicle_count += 1

        # ── RL-Optimizer Calculations ─────────────────
        green_time = compute_green_time(vehicle_count)
        carbon_delta = compute_carbon_saved(vehicle_count, green_time)
        accumulated_carbon = round(accumulated_carbon + carbon_delta * 0.001, 4)

        # ── Draw Zone + HUD ───────────────────────────
        frame = draw_zone(frame, ZONE)
        frame = draw_hud(frame, vehicle_count, green_time, accumulated_carbon, encroachment)

        # ── Encode Frame to JPEG ──────────────────────
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # ── Update Shared State ───────────────────────
        with state_lock:
            state["vehicle_count"] = vehicle_count
            state["encroachment_alert"] = encroachment
            state["dynamic_green_time"] = green_time
            state["carbon_saved_kg"] = accumulated_carbon
            state["frame"] = jpeg.tobytes()

    if cap:
        cap.release()


def generate_synthetic_frame(frame_idx: int) -> np.ndarray:
    """
    Generate a synthetic traffic-like frame when no video file is present.
    Moves colored rectangles to simulate vehicles on a road.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw road background
    cv2.rectangle(frame, (0, 0), (640, 480), (25, 25, 35), -1)
    # Road lanes
    cv2.rectangle(frame, (80, 100), (560, 420), (45, 45, 55), -1)
    for x in range(200, 500, 60):
        cv2.line(frame, (x, 260), (x + 30, 260), (200, 200, 100), 2)

    # Simulate 3–7 moving vehicles
    np.random.seed(frame_idx % 300)
    n = np.random.randint(3, 8)
    for i in range(n):
        vx = (frame_idx * (5 + i * 2) + i * 90) % 580
        vy = 120 + i * 50
        w, h = 60, 30
        cv2.rectangle(frame, (vx, vy), (vx + w, vy + h), (80, 180, 80), -1)
        cv2.rectangle(frame, (vx, vy), (vx + w, vy + h), (0, 255, 100), 1)

    return frame


# ─────────────────────────────────────────────
# Startup Event – Launch Background Thread
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=video_processing_loop, daemon=True)
    thread.start()


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

def frame_generator():
    """Yield MJPEG frames for multipart streaming."""
    while True:
        with state_lock:
            frame_bytes = state.get("frame")

        if frame_bytes is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )
        time.sleep(0.04)  # ~25 FPS cap


@app.get("/api/video-feed", summary="MJPEG Live CCTV Stream")
def video_feed():
    """
    Returns a multipart/x-mixed-replace stream of YOLO-processed frames.
    Suitable for direct use as <img src="http://localhost:8000/api/video-feed" />.
    """
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/traffic-data", summary="Real-Time Traffic Intelligence")
def traffic_data():
    """
    Returns current AI-computed traffic metrics as JSON.
    Polled by frontend every second.
    """
    with state_lock:
        return JSONResponse({
            "vehicle_count": state["vehicle_count"],
            "encroachment_alert": state["encroachment_alert"],
            "dynamic_green_time": state["dynamic_green_time"],
            "carbon_saved_kg": state["carbon_saved_kg"],
            "backend_status": "live" if state["running"] else "initializing",
        })


@app.get("/", summary="Health Check")
def root():
    return {"status": "SMC-Niyantran API is running", "version": "1.0.0"}
