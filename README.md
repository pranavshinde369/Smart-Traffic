<<<<<<< HEAD
# SMC-Niyantran 🚦
### Solapur Mobility Control & Niyantran
> AI-powered Smart City Traffic Platform for Solapur Municipal Corporation

---

## Project Structure

```
SMC_Niyantran_Project/
├── backend/
│   ├── main.py               ← FastAPI + YOLOv8 + OpenCV engine
│   ├── requirements.txt      ← Python dependencies
│   └── traffic_feed.mp4      ← (Drop your traffic video here)
└── frontend/                 ← React + Vite (scaffold with Lovable)
```

---

## Backend Setup & Run

```bash
# 1. Navigate to backend
cd SMC_Niyantran_Project/backend

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your traffic video
# Drop any traffic video as: traffic_feed.mp4
# If missing, backend auto-runs a synthetic demo feed

# 5. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**API will be live at:** `http://localhost:8000`

| Endpoint | Description |
|---|---|
| `GET /api/video-feed` | MJPEG stream (use as `<img src=...>`) |
| `GET /api/traffic-data` | Live JSON metrics |
| `GET /docs` | Auto-generated Swagger UI |

---

## API Response Example

```json
{
  "vehicle_count": 7,
  "encroachment_alert": true,
  "dynamic_green_time": 21,
  "carbon_saved_kg": 0.0117,
  "backend_status": "live"
}
```

---

## Key Logic

### Virtual Geofencing (Encroachment Shield)
Zone defined as `ZONE = (50, 200, 350, 450)` — any vehicle with center point inside this rectangle triggers:
- RED bounding box on frame
- `encroachment_alert: true` in API
- Frontend flashes alert + "E-Challan Drafted" banner

### RL-Optimizer (Green Signal)
```
dynamic_green_time = clamp(vehicle_count × 3, min=20s, max=120s)
```

### Eco-Niyantran (Carbon Saved)
```
time_saved = max(0, 60 - dynamic_green_time)   # vs. legacy fixed 60s
carbon_saved = vehicle_count × time_saved × 0.0028 kg/vehicle/sec
```

---

## Notes
- YOLOv8n model (`yolov8n.pt`) is ~6MB and downloads automatically on first run
- Backend uses a daemon thread for video processing — never blocks API
- If `traffic_feed.mp4` is missing, a synthetic moving-vehicle simulation runs automatically
- CORS is open (`*`) for local development
