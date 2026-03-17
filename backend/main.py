"""
SMC-Niyantran Backend  v2.0
Solapur Mobility Control & Niyantran
FastAPI + YOLOv8 + OpenCV Traffic Intelligence Engine

New in v2.0:
  - Alert log with auto E-Challan numbers  → GET /api/alert-log
  - 30-day carbon history array            → GET /api/carbon-log
  - Live-configurable settings             → POST /api/settings
  - System stats (FPS, uptime, frames)     → GET /api/system-stats
  - Yatra Mode signal override             → via POST /api/settings
  - Hot-reload: ZONE + confidence from settings (no restart needed)
"""

import cv2
import time
import uuid
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from ultralytics import YOLO
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ─────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────
app = FastAPI(
    title="SMC-Niyantran API",
    description="Solapur Mobility Control – AI Traffic Engine v2.0",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

VEHICLE_CLASSES   = [2, 3, 5, 7]           # car, motorcycle, bus, truck
VIDEO_PATH        = Path(__file__).parent / "traffic_feed.mp4"
MODEL_PATH        = "yolov8n.pt"
CARBON_PER_VEH_SEC = 0.0028                # kg CO2 per vehicle per idle-second saved
BASELINE_GREEN    = 60                     # legacy fixed signal (seconds)
CLASS_NAMES       = {2: "CAR", 3: "BIKE", 5: "BUS", 7: "TRUCK"}
ZONE_NAMES        = {
    "market_yard":  "Market Yard Junction",
    "navi_peth":    "Navi Peth Crossing",
    "station_road": "Station Road",
    "hutatma":      "Hutatma Chowk",
}

# ─────────────────────────────────────────────
# Live-Configurable Settings  (hot-reloaded by processing thread)
# ─────────────────────────────────────────────
settings = {
    # No-parking zone pixel coords on 640×480 frame
    "zone_x1":          50,
    "zone_y1":         200,
    "zone_x2":         350,
    "zone_y2":         450,
    # YOLO confidence threshold
    "confidence":      0.35,
    # Yatra Mode – overrides RL signal with fixed value
    "yatra_mode":      False,
    "yatra_green_time": 45,     # seconds (fixed override during Yatra)
    # Active junction label shown in HUD
    "active_junction": "Market Yard Junction",
}
settings_lock = threading.Lock()

# ─────────────────────────────────────────────
# Shared Runtime State
# ─────────────────────────────────────────────
state = {
    "vehicle_count":      0,
    "encroachment_alert": False,
    "dynamic_green_time": 20,
    "carbon_saved_kg":    0.0,
    "frame":              None,     # Latest JPEG bytes
    "running":            False,
    # System stats
    "fps":                0.0,
    "frame_count":        0,
    "start_time":         None,
    "uptime_seconds":     0,
}
state_lock = threading.Lock()

# ─────────────────────────────────────────────
# Alert Log  (max 100 entries, newest first)
# ─────────────────────────────────────────────
alert_log   = deque(maxlen=100)
alert_lock  = threading.Lock()
_challan_counter = 0
_challan_lock    = threading.Lock()

def generate_challan_number() -> str:
    """Generate unique E-Challan ID: SMC-YYYYMMDD-XXXX"""
    global _challan_counter
    with _challan_lock:
        _challan_counter += 1
        date_str = datetime.now().strftime("%Y%m%d")
        return f"SMC-{date_str}-{_challan_counter:04d}"

def log_alert(zone: str, vehicle_count: int, status: str = "VIOLATION"):
    """Append an alert entry to the alert log."""
    entry = {
        "id":            str(uuid.uuid4())[:8],
        "challan_no":    generate_challan_number() if status == "VIOLATION" else None,
        "timestamp":     datetime.now().isoformat(),
        "timestamp_fmt": datetime.now().strftime("%d/%m/%Y, %I:%M:%S %p"),
        "zone":          zone,
        "status":        status,
        "vehicle_count": vehicle_count,
        "fine_inr":      500 if status == "VIOLATION" else 0,
        "details":       "Encroachment detected in No-Parking Zone" if status == "VIOLATION"
                         else "Zone cleared — no violations",
    }
    with alert_lock:
        alert_log.appendleft(entry)

# ─────────────────────────────────────────────
# Carbon History  (30-day rolling array, updated every ~60s)
# ─────────────────────────────────────────────
carbon_history = deque(maxlen=30)   # each entry: {"day": "Day N", "carbon_kg": float, "wait_time": int}
carbon_lock    = threading.Lock()
_last_carbon_update = 0.0

def _seed_carbon_history():
    """
    Pre-populate 30-day history with a realistic RL learning curve:
    wait time drops from ~80s (Day 1) to ~25s (Day 30) as RL converges.
    Carbon saved increases as signal efficiency improves.
    """
    np.random.seed(42)
    with carbon_lock:
        carbon_history.clear()
        for i in range(30):
            # Exponential decay from 80 to 25 with small noise
            wait = int(80 * np.exp(-0.055 * i) + 25 + np.random.uniform(-3, 3))
            wait = max(25, min(85, wait))
            # Carbon saved increases as wait time reduces (more efficient signaling)
            efficiency = (80 - wait) / 55           # 0 → 1 as wait drops 80→25
            carbon = round(4.2 * efficiency + np.random.uniform(0, 0.4), 2)
            carbon_history.append({
                "day":        f"Day {i + 1}",
                "wait_time":  wait,
                "carbon_kg":  carbon,
            })

# ─────────────────────────────────────────────
# Helper: RL Optimizer
# ─────────────────────────────────────────────

def compute_green_time(count: int) -> int:
    """Dynamic green signal: clamp(count × 3, 20s, 120s)"""
    return min(120, max(20, count * 3))

def compute_carbon_saved(count: int, green_time: int) -> float:
    """CO2 saved vs. 60s legacy baseline."""
    time_saved = max(0, BASELINE_GREEN - green_time)
    return round(count * time_saved * CARBON_PER_VEH_SEC, 4)

# ─────────────────────────────────────────────
# Helper: Frame Drawing
# ─────────────────────────────────────────────

def get_zone() -> tuple:
    with settings_lock:
        return (
            settings["zone_x1"], settings["zone_y1"],
            settings["zone_x2"], settings["zone_y2"],
        )

def is_in_zone(cx: int, cy: int, zone: tuple) -> bool:
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2

def draw_zone(frame: np.ndarray, zone: tuple) -> np.ndarray:
    x1, y1, x2, y2 = zone
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 100, 255), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 160, 255), 2)
    cv2.putText(frame, "NO PARKING ZONE",
                (x1 + 4, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    return frame

def draw_hud(frame: np.ndarray, count: int, green_time: int,
             carbon: float, alert: bool, yatra: bool, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]

    # Top bar color: amber during Yatra, dark navy normally
    bar_color = (20, 100, 200) if yatra else (10, 10, 30)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), bar_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    title = "YATRA MODE ACTIVE" if yatra else "SMC-NIYANTRAN"
    title_color = (0, 200, 255) if yatra else (0, 220, 180)
    cv2.putText(frame, f"{title} | MARKET YARD JN.",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, title_color, 1, cv2.LINE_AA)

    # Stats row
    stats = f"Vehicles: {count}  |  Green: {green_time}s  |  CO2 Saved: {carbon}kg  |  FPS: {fps:.1f}"
    cv2.putText(frame, stats,
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # Yatra badge (top-right)
    if yatra:
        cv2.rectangle(frame, (w - 130, 5), (w - 5, 30), (0, 140, 255), -1)
        cv2.putText(frame, "YATRA OVERRIDE",
                    (w - 126, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Flashing encroachment alert bar at bottom
    if alert:
        tick = int(time.time() * 2) % 2
        if tick == 0:
            cv2.rectangle(frame, (0, h - 32), (w, h), (0, 0, 180), -1)
            cv2.putText(frame, "!! ENCROACHMENT ALERT – E-CHALLAN DRAFTED !!",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

# ─────────────────────────────────────────────
# Synthetic Frame Generator (no video fallback)
# ─────────────────────────────────────────────

def generate_synthetic_frame(frame_idx: int) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0),   (640, 480), (22, 22, 32), -1)   # bg
    cv2.rectangle(frame, (80, 90), (560, 420), (42, 42, 52), -1)   # road
    # Lane markings
    for x in range(160, 520, 55):
        cv2.line(frame, (x, 255), (x + 28, 255), (180, 180, 80), 2)
    # Moving vehicles
    np.random.seed(frame_idx % 400)
    n = np.random.randint(4, 9)
    colors = [(80, 180, 80), (80, 120, 200), (200, 120, 60), (160, 80, 200)]
    for i in range(n):
        vx = (frame_idx * (4 + i * 2) + i * 85) % 570
        vy = 110 + (i % 5) * 55
        vw, vh = (70 if i % 3 == 0 else 55), (34 if i % 3 == 0 else 28)
        c = colors[i % len(colors)]
        cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), c, -1)
        cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (255, 255, 255), 1)
    return frame

# ─────────────────────────────────────────────
# Background Video Processing Thread
# ─────────────────────────────────────────────

def video_processing_loop():
    global _last_carbon_update

    model = YOLO(MODEL_PATH)

    use_synthetic = not VIDEO_PATH.exists()
    cap = None
    if not use_synthetic:
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        if not cap.isOpened():
            use_synthetic = True

    with state_lock:
        state["running"]    = True
        state["start_time"] = time.time()

    accumulated_carbon = 0.0
    frame_idx          = 0
    fps_counter        = 0
    fps_timer          = time.time()
    _prev_alert        = False     # tracks alert edge (False→True) for logging

    while True:
        loop_start = time.time()

        # ── 1. Grab Frame ────────────────────────────
        if use_synthetic:
            frame = generate_synthetic_frame(frame_idx)
            frame_idx += 1
            time.sleep(0.05)
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (640, 480))

        # ── 2. Read live settings ────────────────────
        with settings_lock:
            conf_thresh = settings["confidence"]
            yatra_mode  = settings["yatra_mode"]
            yatra_gt    = settings["yatra_green_time"]
            junction    = settings["active_junction"]
        zone = get_zone()

        # ── 3. YOLOv8 Inference ──────────────────────
        results = model(frame, classes=VEHICLE_CLASSES,
                        conf=conf_thresh, verbose=False)

        vehicle_count = 0
        encroachment  = False

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx      = (x1 + x2) // 2
                cy      = (y1 + y2) // 2
                cls_id  = int(box.cls[0])
                conf_v  = float(box.conf[0])
                in_zone = is_in_zone(cx, cy, zone)

                if in_zone:
                    encroachment = True
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 100)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                lbl = f"{CLASS_NAMES.get(cls_id, 'VEH')} {conf_v:.2f}"
                cv2.putText(frame, lbl, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
                vehicle_count += 1

        # ── 4. Alert Edge Detection → log on rising edge ──
        if encroachment and not _prev_alert:
            log_alert(junction, vehicle_count, status="VIOLATION")
        elif not encroachment and _prev_alert:
            log_alert(junction, vehicle_count, status="CLEAR")
        _prev_alert = encroachment

        # ── 5. RL Optimizer / Yatra Override ────────
        if yatra_mode:
            green_time = yatra_gt
        else:
            green_time = compute_green_time(vehicle_count)

        # ── 6. Carbon Accumulation ───────────────────
        carbon_delta    = compute_carbon_saved(vehicle_count, green_time)
        accumulated_carbon = round(accumulated_carbon + carbon_delta * 0.001, 4)

        # Update 30-day carbon history every ~60 seconds
        now = time.time()
        if now - _last_carbon_update >= 60:
            _last_carbon_update = now
            day_label = f"Day {len(carbon_history) + 1}"
            rl_wait   = max(25, green_time - vehicle_count)   # simulated learning output
            with carbon_lock:
                carbon_history.append({
                    "day":       day_label,
                    "wait_time": rl_wait,
                    "carbon_kg": round(accumulated_carbon, 3),
                })

        # ── 7. FPS Counter ───────────────────────────
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            current_fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer   = time.time()
            with state_lock:
                state["fps"] = round(current_fps, 1)

        # ── 8. Draw Overlays ─────────────────────────
        frame = draw_zone(frame, zone)
        with state_lock:
            current_fps = state["fps"]
        frame = draw_hud(frame, vehicle_count, green_time,
                         accumulated_carbon, encroachment, yatra_mode, current_fps)

        # ── 9. JPEG Encode ───────────────────────────
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])

        # ── 10. Write Shared State ───────────────────
        with state_lock:
            state["vehicle_count"]      = vehicle_count
            state["encroachment_alert"] = encroachment
            state["dynamic_green_time"] = green_time
            state["carbon_saved_kg"]    = accumulated_carbon
            state["frame"]              = jpeg.tobytes()
            state["frame_count"]       += 1
            state["uptime_seconds"]     = int(time.time() - state["start_time"])

    if cap:
        cap.release()

# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    _seed_carbon_history()   # pre-populate 30-day RL learning curve
    thread = threading.Thread(target=video_processing_loop, daemon=True)
    thread.start()

# ─────────────────────────────────────────────
# MJPEG Frame Generator
# ─────────────────────────────────────────────

def frame_generator():
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
        time.sleep(0.04)   # ~25 FPS cap

# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/", summary="Health Check")
def root():
    return {"status": "SMC-Niyantran API v2.0 running", "version": "2.0.0"}


@app.get("/api/video-feed", summary="MJPEG Live CCTV Stream")
def video_feed():
    """
    multipart/x-mixed-replace MJPEG stream.
    Use as: <img src="http://localhost:8000/api/video-feed" />
    """
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/traffic-data", summary="Real-Time Traffic Metrics")
def traffic_data():
    """
    Polled by frontend every 1000ms.
    Returns vehicle_count, encroachment_alert, dynamic_green_time,
    carbon_saved_kg, yatra_mode, backend_status.
    """
    with state_lock:
        sv = dict(state)
    with settings_lock:
        ym = settings["yatra_mode"]
        ygt = settings["yatra_green_time"]
    return JSONResponse({
        "vehicle_count":      sv["vehicle_count"],
        "encroachment_alert": sv["encroachment_alert"],
        "dynamic_green_time": sv["dynamic_green_time"],
        "carbon_saved_kg":    sv["carbon_saved_kg"],
        "yatra_mode":         ym,
        "yatra_green_time":   ygt,
        "fps":                sv["fps"],
        "backend_status":     "live" if sv["running"] else "initializing",
    })


@app.get("/api/alert-log", summary="Encroachment Alert History")
def get_alert_log():
    """
    Returns list of alert events (newest first, max 100).
    Used by /alerts page table.
    Each entry: challan_no, timestamp, zone, status, vehicle_count, fine_inr.
    """
    with alert_lock:
        logs = list(alert_log)
    return JSONResponse({"alerts": logs, "total": len(logs)})


@app.get("/api/carbon-log", summary="30-Day Carbon & RL History")
def get_carbon_log():
    """
    Returns 30-day rolling array for Analytics charts.
    Each entry: day, wait_time (RL learning curve), carbon_kg.
    """
    with carbon_lock:
        history = list(carbon_history)
    return JSONResponse({"history": history, "days": len(history)})


@app.get("/api/system-stats", summary="System Performance Stats")
def system_stats():
    """
    Used by Camera Feed page stats bar.
    Returns fps, frame_count, uptime_seconds, resolution.
    """
    with state_lock:
        return JSONResponse({
            "fps":             state["fps"],
            "frame_count":     state["frame_count"],
            "uptime_seconds":  state["uptime_seconds"],
            "resolution":      "640x480",
            "model":           "YOLOv8n",
            "vehicle_classes": ["car", "motorcycle", "bus", "truck"],
        })


@app.post("/api/settings", summary="Update Live Settings")
def update_settings(body: dict = Body(...)):
    """
    Hot-reload settings without restarting the server.
    Accepted fields:
      zone_x1, zone_y1, zone_x2, zone_y2  (int, pixel coords on 640×480)
      confidence                           (float, 0.1–0.9)
      yatra_mode                           (bool)
      yatra_green_time                     (int, seconds)
      active_junction                      (str)

    Example body:
      {"yatra_mode": true, "yatra_green_time": 45}
      {"zone_x1": 50, "zone_y1": 150, "zone_x2": 300, "zone_y2": 400, "confidence": 0.45}
    """
    allowed = {
        "zone_x1", "zone_y1", "zone_x2", "zone_y2",
        "confidence", "yatra_mode", "yatra_green_time", "active_junction",
    }
    updated = {}
    with settings_lock:
        for key, val in body.items():
            if key in allowed:
                settings[key] = val
                updated[key]  = val
    return JSONResponse({
        "status":  "ok",
        "updated": updated,
        "current": dict(settings),
    })


@app.get("/api/settings", summary="Get Current Settings")
def get_settings():
    """Returns current live settings. Used by Settings page to populate fields."""
    with settings_lock:
        return JSONResponse(dict(settings))
